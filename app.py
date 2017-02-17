from flask import Flask, render_template, request
import centCom
import os

app = Flask(__name__)

@app.route("/")
def main():
	re = str(request.args.get('getinput'))

	print("Opening model...")

	classifier = centCom('initial_model.h5')
	classifier.preprocess()
	classifier.all_tokenize()

	print("Predicting...")
	results = classifier.predict(re)


	str(results)
	max_indices = np.argmax(results, axis=1)








	return results

if __name__ == "__main__":
	port = int(os.environ.get('PORT', 5000))
	app.run(host="0.0.0.0", port=port, debug=True)
	#  