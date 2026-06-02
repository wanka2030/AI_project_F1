import joblib
import os
import pandas as pd
from flask import Flask, request, render_template

current_dir = os.path.dirname(os.path.abspath(__file__))
# 組合出正確的模型檔案路徑
model_path = os.path.join(current_dir, "F1.pkl")

# 載入模型
model_pretrained = joblib.load(model_path)

app = Flask(__name__)

@app.route("/")
def formPage():
    return render_template("form.html")

@app.route("/submit", methods=["POST"])
def submit():
    if request.method == "POST":
        form_data = request.form
        
        # 將取得的表單資料轉換成浮點數格式
        tyre_life = float(form_data["TyreLife"])
        stint = float(form_data["Stint"])
        lap_number = float(form_data["LapNumber"])
        position = float(form_data["Position"])
        laps_since_last_pit = float(form_data["Laps_Since_LastPit"])

        # 整理成 DataFrame 所需的二維陣列
        data = [[
            tyre_life,
            stint,
            lap_number,
            position,
            laps_since_last_pit
        ]]

        df = pd.DataFrame(
            data,
            columns=[
                "TyreLife",
                "Stint",
                "LapNumber",
                "Position",
                "Laps_Since_LastPit"
            ],
        )
        
        # 進行預測與取得機率
        result = model_pretrained.predict(df)
        result_proba = model_pretrained.predict_proba(df)
        
        print(f"Result: {result}")
        print(f"Result Probabilities: {result_proba}")
        
        # 判斷結果 (假設 1 代表進站，0 代表繼續跑)
        if result[0] == 1:
            prediction = f"即將進站 (Y) - 系統信心 {result_proba[0][1]:.5f}"
        else:
            prediction = f"繼續留在場上 (N) - 系統信心 {result_proba[0][0]:.5f}"
            
        return render_template(
            "form.html",
            TyreLife=form_data["TyreLife"],
            Stint=form_data["Stint"],
            LapNumber=form_data["LapNumber"],
            Position=form_data["Position"],
            Laps_Since_LastPit=form_data["Laps_Since_LastPit"],
            prediction=prediction,
        )

if __name__ == "__main__":
    # 加上 debug=True 方便開發時隨時看到錯誤訊息
    app.run(debug=True)