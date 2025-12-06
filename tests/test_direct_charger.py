"""
Direct charger visibility test - teleport robot to see charger
"""

import os
import sys
import numpy as np

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)


def main():
    print("="*60)
    print("DIRECT CHARGER VISIBILITY TEST")
    print("Teleporting robot to positions where charger MUST be visible")
    print("="*60)
    
    from omni.isaac.kit import SimulationApp
    simulation_app = SimulationApp({"headless": False, "width": 1280, "height": 720})
    
    try:
        from omni.isaac.core import World
        from omni.isaac.core.objects import FixedCuboid
        from omni.isaac.core.utils.stage import add_reference_to_stage
        from omni.isaac.wheeled_robots.robots import WheeledRobot
        from pxr import UsdLux, UsdGeom, Gf
        from PIL import Image, ImageDraw
        import omni.kit.viewport.utility as viewport_utils
        import tempfile
        
        print("\n[1/5] Creating simple test scene...")
        world = World(stage_units_in_meters=1.0)
        world.scene.add_default_ground_plane()
        
        # Lighting
        stage = world.stage
        UsdLux.DistantLight.Define(stage, "/World/sunLight").CreateIntensityAttr(1000.0)
        UsdLux.DomeLight.Define(stage, "/World/domeLight").CreateIntensityAttr(300.0)
        
        # Only charger - no plant blocking!
        print("\n[2/5] Creating GREEN charger (no plant)...")
        charger = world.scene.add(
            FixedCuboid(
                prim_path="/World/charger",
                name="charger",
                position=np.array([2.0, 0.0, 0.25]),  # Closer, lower
                scale=np.array([0.5, 0.5, 0.5]),  # Bigger
                size=1.0,
                color=np.array([0.0, 1.0, 0.0]),  # BRIGHT GREEN
            )
        )
        print("  Charger: position=(2.0, 0.0, 0.25), color=BRIGHT GREEN")
        
        # Also add a RED box for comparison
        plant = world.scene.add(
            FixedCuboid(
                prim_path="/World/plant",
                name="plant",
                position=np.array([2.0, 1.5, 0.25]),  # To the side
                scale=np.array([0.5, 0.5, 0.5]),
                size=1.0,
                color=np.array([1.0, 0.0, 0.0]),  # BRIGHT RED
            )
        )
        print("  Plant: position=(2.0, 1.5, 0.25), color=BRIGHT RED")
        
        # TurtleBot
        print("\n[3/5] Spawning TurtleBot...")
        turtlebot_usd = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/Robots/Turtlebot/Turtlebot3/turtlebot3_burger.usd"
        add_reference_to_stage(usd_path=turtlebot_usd, prim_path="/World/Turtlebot")
        
        robot = world.scene.add(
            WheeledRobot(
                prim_path="/World/Turtlebot",
                name="turtlebot",
                wheel_dof_names=["wheel_left_joint", "wheel_right_joint"],
                create_robot=False,
                position=np.array([0.0, 0.0, 0.0]),
            )
        )
        
        # Camera
        print("\n[4/5] Creating camera...")
        camera_path = "/World/Turtlebot/base_link/robot_camera"
        camera_prim = UsdGeom.Camera.Define(stage, camera_path)
        camera_prim.CreateFocalLengthAttr(24.0)
        camera_prim.CreateHorizontalApertureAttr(20.955)
        camera_prim.CreateClippingRangeAttr(Gf.Vec2f(0.01, 100.0))
        
        xformable = UsdGeom.Xformable(camera_prim.GetPrim())
        xformable.ClearXformOpOrder()
        xformable.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.20))
        xformable.AddRotateXYZOp().Set(Gf.Vec3f(90.0, 0.0, -90.0))
        
        world.reset()
        
        # Settle
        for _ in range(50):
            world.step(render=True)
        
        print("\n[5/5] Testing perception...")
        from perception.vision_model import PerceptionModule
        
        perception = PerceptionModule(
            model_name="IDEA-Research/grounding-dino-tiny",
            text_prompts={
                'charger': 'green box . green cube . green object',
                'plant': 'red box . red cube . red object',
            },
            temperature=2.0,
            threshold=0.2,  # Lower threshold
        )
        
        # Capture image
        def get_rgb():
            viewport = viewport_utils.get_active_viewport()
            viewport.set_active_camera(camera_path)
            for _ in range(5):
                world.step(render=True)
            
            import omni.kit.viewport.utility.capture as capture
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
                temp_path = f.name
            
            cap = capture.capture_viewport_to_file(viewport, temp_path)
            while not cap.is_finished():
                world.step(render=True)
            
            from PIL import Image
            img = Image.open(temp_path).convert('RGB')
            img = img.resize((640, 480))
            rgb = np.array(img)
            os.unlink(temp_path)
            return rgb
        
        # Test
        print("\n" + "="*60)
        print("Robot looking at GREEN charger and RED plant")
        print("="*60)
        
        rgb = get_rgb()
        robot_pose = (0.0, 0.0, 0.0)
        detections = perception.detect(rgb, robot_pose)
        
        print(f"\nDetections: {len(detections)}")
        for det in detections:
            print(f"  {det['class']}: conf={det['confidence']:.2f}, color_score={det['color_score']:.2f}")
        
        # Analyze image colors
        print("\n--- Image Color Analysis ---")
        h, w = rgb.shape[:2]
        center_region = rgb[h//4:3*h//4, w//4:3*w//4]
        
        r_mean = np.mean(center_region[:,:,0])
        g_mean = np.mean(center_region[:,:,1])
        b_mean = np.mean(center_region[:,:,2])
        print(f"Center region avg RGB: ({r_mean:.1f}, {g_mean:.1f}, {b_mean:.1f})")
        
        # Find green pixels
        green_mask = (rgb[:,:,1] > 100) & (rgb[:,:,1] > rgb[:,:,0] + 30) & (rgb[:,:,1] > rgb[:,:,2] + 30)
        green_percent = np.mean(green_mask) * 100
        print(f"Green pixels in image: {green_percent:.1f}%")
        
        # Find red pixels
        red_mask = (rgb[:,:,0] > 100) & (rgb[:,:,0] > rgb[:,:,1] + 30) & (rgb[:,:,0] > rgb[:,:,2] + 30)
        red_percent = np.mean(red_mask) * 100
        print(f"Red pixels in image: {red_percent:.1f}%")
        
        # Save image
        img = Image.fromarray(rgb)
        draw = ImageDraw.Draw(img)
        colors = {'charger': (0, 255, 0), 'plant': (255, 0, 0)}
        for det in detections:
            x1, y1, x2, y2 = det['bbox_2d']
            color = colors.get(det['class'], (255, 255, 0))
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            label = f"{det['class']}: {det['confidence']:.2f}"
            draw.text((x1, max(5, y1-15)), label, fill=color)
        
        save_path = os.path.join(project_root, "direct_charger_test.png")
        img.save(save_path)
        print(f"\nSaved: {save_path}")
        
        if not detections:
            print("\n⚠ NO DETECTIONS - Check the saved image to see what robot sees!")
        
        print("\nKeep window open - Ctrl+C to exit...")
        while simulation_app.is_running():
            world.step(render=True)
            
    except KeyboardInterrupt:
        print("\nStopping...")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
