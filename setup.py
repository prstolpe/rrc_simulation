from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

d = generate_distutils_setup(
    packages=[
        "rrc_simulation",
        "rrc_simulation.gym_wrapper",
        "rrc_simulation.gym_wrapper.envs",
        "rrc_simulation.tasks"
    ],
    package_dir={"": "python"},
    package_data={
        "": [
            "robot_properties_fingers/meshes/stl/*",
            "robot_properties_fingers/meshes/stl/edu/*",
            "robot_properties_fingers/meshes/stl/pro/*",
            "robot_properties_fingers/urdf/*",
            "robot_properties_fingers/urdf/pro/*",
            "robot_properties_fingers/urdf/edu/*",
        ]
    },
    scripts=[
        "scripts/replay_action_log.py",
        "scripts/rrc_evaluate",
        "scripts/run_evaluate_policy_all_levels.py",
        "scripts/run_replay_all_levels.py",
    ],
)

setup(**d)
