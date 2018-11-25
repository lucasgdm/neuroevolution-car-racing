# neuroevolution-car-racing

An evolutionary algorithm built using an adapted version of [OpenAI's CarRacing-v0 challenge](https://gym.openai.com/envs/CarRacing-v0/)
<p align="center">
  <a href="https://raw.githubusercontent.com/lucasgdm/evolutionary-neuralnet-car-racing/master/demo.mp4">
    <img src="https://raw.githubusercontent.com/lucasgdm/evolutionary-neuralnet-car-racing/master/demo.gif" />
  </a>
</p>

The multilayer perceptron receives with the following inputs:
- The curvature of the road at N sample points ahead of the car
- Car speed
- Car angle
- Speed direction (might diverge from that of the car when it's skidding)
- Wheel angle
- Car angular velocity
- Distance between the car and the center of the road

## Installing dependencies
Install swig, which is needed for Box2D to work  
`$ apt install swig`

Install pip dependencies, [preferably using a virtual environment](https://docs.python.org/3/tutorial/venv.html)  
`$ pip install -r requirements.txt`

## Running
Render the best neural net  
`$ python3 render.py`

... Or play against it. Use arrow keys to move and press return key to start over. Do not accelerate and steer at the same time  
`$ python3 play_against.py`

... Or run the algorithm  
`$ python3 main.py`

Remove `saved_dnas` if you want the algorithm to start from scratch  
`$ rm saved_dnas; python3 main.py`

