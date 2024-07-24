# Mirror


## Docker Mirror

- [国内的 Docker Hub 镜像加速器，由国内教育机构与各大云服务商提供的镜像加速服务 | Dockerized 实践 https://github.com/y0ngb1n/dockerized](https://gist.github.com/y0ngb1n/7e8f16af3242c7815e7ca2f0833d3ea6)

### Root

edit `/etc/docker/daemon.json`

```json
{
    "registry-mirrors": ["https://docker.m.daocloud.io", "https://dockerhub.azk8s.cn", "https://docker.mirrors.ustc.edu.cn", "https://dockerproxy.com", "https://mirror.baidubce.com", "https://docker.nju.edu.cn", "https://mirror.iscas.ac.cn"]
}
```

```bash
sudo systemctl daemon-reload
sudo systemctl restart docker
```

### Rootless

edit `~/.config/docker/daemon.json`

```bash
systemctl --user daemon-reload
systemctl --user restart docker
```

### Check installation

```bash
$ docker info
...
 Registry Mirrors:
  https://docker.m.daocloud.io/
  https://dockerhub.azk8s.cn/
  https://docker.mirrors.ustc.edu.cn/
  https://dockerproxy.com/
  https://mirror.baidubce.com/
  https://docker.nju.edu.cn/
  https://mirror.iscas.ac.cn/
  ...
```
