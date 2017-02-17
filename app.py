from flask import Flask, render_template, request
app = Flask(__name__)

@app.route("/")
def main():
	re = str(request.args.get('getinput'))
	var ="Hello World!"
	return re

if __name__ == "__main__":
	app.run()