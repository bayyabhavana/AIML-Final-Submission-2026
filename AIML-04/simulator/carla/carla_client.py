
import sys
import time

try:
    import carla
except ImportError:
    print("CARLA Python API is not installed.")
    print("Install CARLA and configure its Python API before running.")
    sys.exit(1)


def main():
    print("Connecting to CARLA simulator...")

    client = carla.Client("localhost", 2000)
    client.set_timeout(5.0)

    try:
        world = client.get_world()

        print("CARLA connection successful!")
        print("Map:", world.get_map().name)

        vehicles = world.get_actors().filter("vehicle.*")
        pedestrians = world.get_actors().filter("walker.*")

        print("Vehicles in simulation:", len(vehicles))
        print("Pedestrians in simulation:", len(pedestrians))

        print("CARLA client test completed.")

    except Exception as error:
        print("CARLA connection failed:", error)
        print("Start the CARLA simulator before running this client.")


if __name__ == "__main__":
    main()
