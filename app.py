"""
VRP Web App — Flask wrapper around vrp_engine.py
"""
import io, traceback
from pathlib import Path
from flask import Flask, request, render_template, jsonify, send_file
from vrp_engine import run_vrp, load_from_excel

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024   # 20 MB

ALLOWED = {"xlsm", "xlsx"}

def _allowed(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/run", methods=["POST"])
def run():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded."}), 400
    f = request.files["file"]
    if f.filename == "" or not _allowed(f.filename):
        return jsonify({"error": "Please upload an .xlsm or .xlsx file."}), 400

    try:
        sheets  = load_from_excel(io.BytesIO(f.read()))
        for sheet in ["locations", "vehicles", "parameters"]:
            if sheets[sheet].empty:
                return jsonify({"error": f"Sheet '{sheet}' is missing or empty."}), 422

        result = run_vrp(sheets)

        if result["status"] == "infeasible":
            return jsonify({"error": "No feasible solution found. "
                            "Check time windows, capacities, and route time limit."}), 422
        if result["status"] == "error":
            return jsonify({"error": result.get("message", "Unknown error.")}), 422

        return jsonify({
            "status":        "ok",
            "total_routes":  result["total_routes"],
            "total_stops":   result["total_stops"],
            "total_time":    result["total_time"],
            "dropped_nodes": result["dropped_nodes"],
            "routes":        result["routes"],
            "csv":           result["csv"],
        })

    except RuntimeError as e:
        # OSRM failure etc.
        return jsonify({"error": str(e)}), 503
    except Exception:
        return jsonify({
            "error":  "Unexpected error. Check your workbook format.",
            "detail": traceback.format_exc(),
        }), 500


@app.route("/sample")
def sample():
    path = Path(__file__).parent / "static" / "VRP_sample.xlsx"
    return send_file(path, as_attachment=True,
                     download_name="VRP_sample.xlsx",
                     mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


if __name__ == "__main__":
    app.run(debug=True, port=5001)
