# Accelerated-Computing

My HW submissions for MIT 6.S894 https://accelerated-computing.academy/fall25/.

# Commands

All commands assume we are in the directory `~/accelerated-computing`

**Setup telerun:**
Clone the repo
```bash
# clone repo
mkdir -p telerun
cd ~
git clone https://github.com/accelerated-computing-class/telerun.git
cp -r telerun/* accelerated-computing/telerun/
rm -rf telerun
cd accelerated-computing

# login
cd telerun
python telerun.py login
cd ..
```

**Clone a new lab:**
```bash
mkdir -p lab1
cd ~
git clone https://github.com/accelerated-computing-class/lab1.git
cp -r lab1/* accelerated-computing/lab1/
rm -rf lab1
cd accelerated-computing
```

**Submit to telerun:**
```bash
python telerun/telerun.py submit lab1/mandelbrot_cpu.cpp
```
