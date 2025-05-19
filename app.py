from flask import Flask, request, render_template
import mlflow.sklearn
import numpy as np

# Configuración Flask
app = Flask(__name__)
mlflow.set_tracking_uri("http://localhost:9090")
mlflow.set_registry_uri("http://localhost:9090")
# Ruta principal
@app.route("/", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        try:
            # Obtener datos del formulario
            Pclass = int(request.form["Pclass"])
            Sex    = int(request.form["Sex"])
            Age    = float(request.form["Age"])
            SibSp  = int(request.form["SibSp"])
            Parch  = int(request.form["Parch"])
            Fare   = float(request.form["Fare"])

            # Crear vector de entrada
            features = np.array([[Pclass, Sex, Age, SibSp, Parch, Fare]])

            # Cargar el modelo desde MLflow
            model = mlflow.sklearn.load_model("models:/titanic_v1/2")

            # Realizar predicción
            pred = model.predict(features)[0]

            result = "Sobrevive ✅" if pred == 1 else "No sobrevive ❌"
            return render_template("index.html", prediction=result)

        except Exception as e:
            return f"❌ Error: {str(e)}"

    return render_template("index.html", prediction=None)

# Correr localmente
if __name__ == "__main__":
    app.run(debug=True)
