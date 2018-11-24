# evolutionary-neuralnet-car-racing

An evolutionary algorithm built using an adapted version of [OpenAI's CarRacing-v0 challenge](https://gym.openai.com/envs/CarRacing-v0/)
<p align="center">
  <a href="https://raw.githubusercontent.com/lucasgdm/evolutionary-neuralnet-car-racing/master/demo.mp4">
    <img src="https://raw.githubusercontent.com/lucasgdm/evolutionary-neuralnet-car-racing/master/demo.gif" />
  </a>
</p>


## Instructions
Install swig (needed for Box2D to work)  
`$ apt install swig`

Install pip dependencies ([preferably using a virtual environment](https://docs.python.org/3/tutorial/venv.html))  
`$ pip install -r requirements.txt`

Render the best neural net  
`$ python3 render.py`

... Or play against it (use arrow keys to move and press return to start over. Do not accelerate and steer at the same time)  
`$ python3 play_against.py`

... Or run the algorithm  
`$ python3 main.py`

