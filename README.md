# API detect bank's logo

### Installing

Download Pre-trained model from google drive:https://drive.google.com/file/d/1DThZRsLugRoY3os81Gcl-CEjs5lJfZna/view?usp=sharing

```
pip install -r requirements.txt
```

## Running the app
Start Flask app
```
python app.py
```
wait until this appear 
```
Running on http://127.0.0.1:5000/
```
Everything is ready

### Test

Go to cd folder and test
curl -X POST -F image=@<image.jpg> "http://localhost:5000/predict"

example
```
cd "test data"
curl -X POST -F image=@ag.jpg "http://localhost:5000/predict"

reponse go like this
{
  "predictions": [
    {
      "AGRIBANK": 2,
      "password-form": 1
    }
  ],
  "success": true
}
```
## Authors

<<<<<<< HEAD
* **Grey Matter** 
=======
* **Grey Matter** 
>>>>>>> a2183cddac676c5ca7049133698e450c3d31b81c
