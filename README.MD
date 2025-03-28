# ELECTRE IV - Decision Making App

This project is a **Streamlit** application that implements the **ELECTRE IV** multi-criteria decision-making method. The app allows users to input alternatives, criteria, and weights, and calculates rankings based on concordance and discordance dominance matrices.

---

## 🚀 Features
- **User-Friendly Input:** Simple UI for adding alternatives, criteria, and weights.
- **Mobile & Desktop Support:** Optimized layout for different screen sizes.
- **Concordance & Discordance Matrices:** Calculates dominance matrices.
- **Handles Ties in Ranking:** Displays results properly when alternatives have equal ranks.
- **Dynamic Results Display:** Users can view either the best alternative or a full ranking list.

---

## 📌 Installation
1. **Clone this repository:**
   ```sh
   git clone https://github.com/TMTMPST/SPK-Electre-Streamlit.git
   cd electre-streamlit
   ```

2. **Create a virtual environment (Optional but recommended):**
   ```sh
   python -m venv venv
   source venv/bin/activate  # On macOS/Linux
   venv\Scripts\activate     # On Windows
   ```

3. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

---

## 🎯 Usage
Run the application with:
```sh
streamlit run app.py
```

Then open the provided **localhost URL** in your browser.

---

## 🛠 Configuration
- You can modify the number of **alternatives** and **criteria** dynamically.
- Adjust weights for each criterion using the sidebar input.
- Results are displayed interactively in a data table.

---

## 📊 Example Output
- **When there’s a clear winner:** The best alternative is highlighted.
- **When there’s a tie:** The app displays multiple top-ranked alternatives.
- **Full ranking option:** Users can toggle between ranked results.

---

## 📜 License
This project is open-source under the **MIT License**.

---

## 🤝 Contributing
Feel free to submit issues or pull requests to improve the project!

🚀 Happy coding!

