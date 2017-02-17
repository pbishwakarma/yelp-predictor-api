from flask import Flask, render_template, request
import os

app = Flask(__name__)

@app.route("/")
def main():
	re = str(request.args.get('getinput'))
	var ="Hello World!"
	return re

if __name__ == "__main__":
	app.run(host="0.0.0.0", port=os.getenv("PORT"), debug=True)
	#  