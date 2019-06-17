# ROLE
Raindrop on lens effect


****
## Enviroment
* python 2.7 
* requirements
  * numpy
  * cv2
  * pillow
  * [pyblur](https://github.com/lospooky/pyblur)
or directly use 
```
pip install -r requirement.txt
```

****
## Introduction
This repository is to create the simulation of water drop on the lens
<img src=/Output_image/aachen_000001_000019_leftImg8bit.png
  width=100%>  
****
### Drop generation

To simulate the waterdrop shape, here using one circle and one oval to make the shape.
Give the circle and the oval 128, which can form different gap, and then create the effect of the water droplet surface through the blur.
The concept and the drop after blur are shown below:  
<img src=/resource/drop_architecture.bmp height= "300" width="300">
<img src=/resource/drop.bmp height= "300" width="300">

****
### Collision
To handle the collision, it will check if the center of the drop is occupied or not
if yes, it will merge, otherwise, do nothing.

Before (To clearly show the shape of the raindrops, here darken the edge of each drop )  
<img src=/resource/collision_before.bmp height= "300" width="500">  
After  
<img src=/resource/collision_after.bmp height= "300" width="500">

****
### Edge
Here using the darken background to make the effect, it can be change by using different parameter.
****

#### Usage
```python
from raindrop.dropgenerator import generateDrops

# it will return image in pillow format
# if using cfg["return_label"] = False
output_image = generateDrops(image_path, cfg)

# if using cfg["return_label"] = True
output_image, output_label = generateDrops(image_path, cfg)
```
The exmple is in the [exmple.py](example.py)


### To DO

- [ ] Modify the collision range(should not only use center)
- [ ] Add other types of water drop
- [ ] Adding the effect as raining animation
