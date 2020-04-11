# API detect bank's logo

### Installing

A step by step series of examples that tell you how to get a development env running

Say what the step will be

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

* **Grey Matter** 
