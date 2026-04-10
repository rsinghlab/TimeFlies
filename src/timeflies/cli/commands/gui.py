"""
TimeFlies CLI GUI Command

Contains the web-based graphical user interface launcher.
"""


def gui_command(args) -> int:
    """Launch web-based graphical user interface."""
    try:
        # Try multiple import strategies for different installation methods
        launch_gui = None

        # Strategy 1: Direct import (works for most cases)
        try:
            from timeflies.gui.gradio_launcher import launch_gui
        except ImportError:
            # Strategy 2: Add source directory to path for editable installs
            import sys
            from pathlib import Path

            # Find TimeFlies source directory
            timeflies_src = None
            current_dir = Path.cwd()

            # Check common locations
            possible_locations = [
                current_dir / ".timeflies_src" / "src",  # User installation
                current_dir / "src",  # Development directory
                Path.cwd() / "src",  # Current working directory src
            ]

            # Add path relative to this file if available
            try:
                possible_locations.append(Path(__file__).parent.parent.parent)
            except NameError:
                pass

            for src_path in possible_locations:
                if (src_path / "timeflies" / "gui" / "gradio_launcher.py").exists():
                    timeflies_src = str(src_path)
                    break

            if timeflies_src and timeflies_src not in sys.path:
                sys.path.insert(0, timeflies_src)
                try:
                    from timeflies.gui.gradio_launcher import launch_gui
                except ImportError:
                    pass

        if launch_gui is None:
            raise ImportError("Could not import GUI launcher from any location")

        print("🚀 Starting TimeFlies Web GUI...")
        print(f"📍 Will launch at: http://{args.host}:{args.port}")

        if args.share:
            print("🌐 Creating public URL for remote access...")
            print("⚠️  WARNING: Public URLs can be accessed by anyone!")

        print("💡 Press Ctrl+C to stop the GUI server")
        print()

        launch_gui(
            server_name=args.host,
            server_port=args.port,
            share=args.share,
            debug=args.debug,
        )

        return 0

    except ImportError as e:
        print(f"❌ Error importing GUI modules: {e}")
        print("💡 Solutions:")
        print("   1. Make sure gradio is installed: pip install gradio>=4.0.0")
        print("   2. Try: timeflies update  # Updates dependencies")
        print("   3. Make sure you're in TimeFlies directory with .timeflies_src/")
        return 1
    except KeyboardInterrupt:
        print("\n✅ GUI server stopped by user")
        return 0
    except Exception as e:
        print(f"❌ Failed to start GUI: {e}")
        print("💡 Make sure you're in the TimeFlies virtual environment")
        return 1
