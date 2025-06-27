![Galaxea Lab](./img/R1DVT_FRUIT.jpg)

# Modified Contributors

- [sujit-168](https://github.com/sujit-168)
- [ruoxi521](https://github.com/ruoxi521)

---

# Isaac Lab (Latest)

[![IsaacSim](https://img.shields.io/badge/IsaacSim-4.5-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://docs.python.org/3/whatsnew/3.10.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/20.04/)
[![Windows platform](https://img.shields.io/badge/platform-windows--64-orange.svg)](https://www.microsoft.com/en-us/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
[![Docs status](https://img.shields.io/badge/docs-passing-brightgreen.svg)](https://isaac-sim.github.io/IsaacLab)
[![License](https://img.shields.io/badge/license-BSD--3-yellow.svg)](https://opensource.org/licenses/BSD-3-Clause)


**Isaac Lab** is a unified and modular framework for robot learning that aims to simplify common workflows
in robotics research (such as RL, learning from demonstrations, and motion planning). It is built upon
[NVIDIA Isaac Sim](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html) to leverage the latest
simulation capabilities for photo-realistic scenes and fast and accurate simulation.

Please refer to our [documentation page](https://isaac-sim.github.io/IsaacLab) to learn more about the
installation steps, features, tutorials, and how to set up your project with Isaac Lab.

## Contributing to Isaac Lab

We wholeheartedly welcome contributions from the community to make this framework mature and useful for everyone.
These may happen as bug reports, feature requests, or code contributions. For details, please check our
[contribution guidelines](https://isaac-sim.github.io/IsaacLab/source/refs/contributing.html).

## Usage

### spawn_robot.py
```bash
# spawn the R1 robot and set random joint positions
python3 source/standalone/galaxea/basic/spawn_robot.py
```

![](./img/spawn_robot.png)

### create_scene.py
```bash
# create a scene consisting of object, table, robot and so on
python3 source/standalone/galaxea/basic/creat_scene.py
```

![](./img/create_sence.png)

### follow_target_diff_ik.py
```bash
python3 source/standalone/galaxea/basic/follow_target_diff_ik.py 
```

![](./img/follow_target_diff_ik.png)

### run_diff_ik.py
```bash
python3 source/standalone/galaxea/basic/follow_target_diff_ik.py 
```

![](./img/run_diff_ik.png)

### collect_demos.py
```bash
#start to collected pick and place fruit data via Galaxea R1-DVT robot
python3 source/standalone/galaxea/rule_based_policy/collect_demos.py --enable_cameras
```

![](./img/collect_demo.png)

### collect_demonstrations.py
```bash
python3 source/standalone/workflows/robomimic/collect_demonstrations.py
```

<div style="position: relative; padding-bottom: 56.25%; height: 0;">
  <iframe src="//player.bilibili.com/player.html?isOutside=true&aid=114686213233537&bvid=BV1qtMxzBEtn&cid=30508451661&p=1&autoplay=0" frameborder="no" scrolling="no" 
    style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;"></iframe>
</div>

### lift.py
```bash
python3 source/standalone/galaxea/scripted_policy/lift.py
```

## Troubleshooting

Please see the [troubleshooting](https://isaac-sim.github.io/IsaacLab/source/refs/troubleshooting.html) section for
common fixes or [submit an issue](https://github.com/isaac-sim/IsaacLab/issues).

For issues related to Isaac Sim, we recommend checking its [documentation](https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/overview.html)
or opening a question on its [forums](https://forums.developer.nvidia.com/c/agx-autonomous-machines/isaac/67).

## Support

* Please use GitHub [Discussions](https://github.com/isaac-sim/IsaacLab/discussions) for discussing ideas, asking questions, and requests for new features.
* Github [Issues](https://github.com/isaac-sim/IsaacLab/issues) should only be used to track executable pieces of work with a definite scope and a clear deliverable. These can be fixing bugs, documentation issues, new features, or general updates.

## License

The Isaac Lab framework is released under [BSD-3 License](LICENSE). The license files of its dependencies and assets are present in the [`docs/licenses`](docs/licenses) directory.

## Acknowledgement

Isaac Lab development initiated from the [Orbit](https://isaac-orbit.github.io/) framework. We would appreciate if you would cite it in academic publications as well:

```
@article{mittal2023orbit,
   author={Mittal, Mayank and Yu, Calvin and Yu, Qinxi and Liu, Jingzhou and Rudin, Nikita and Hoeller, David and Yuan, Jia Lin and Singh, Ritvik and Guo, Yunrong and Mazhar, Hammad and Mandlekar, Ajay and Babich, Buck and State, Gavriel and Hutter, Marco and Garg, Animesh},
   journal={IEEE Robotics and Automation Letters},
   title={Orbit: A Unified Simulation Framework for Interactive Robot Learning Environments},
   year={2023},
   volume={8},
   number={6},
   pages={3740-3747},
   doi={10.1109/LRA.2023.3270034}
}
```

- https://github.com/userguide-galaxea/Galaxea_Lab/
- https://userguide-galaxea.github.io/Product_User_Guide/