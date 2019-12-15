from flask import Flask 

app = Flask(__name__)

@app.route('/')

def hell0():
	return "hello world"

