# 🧾✨ Forgeo – AI Document Authenticator

> **Forget forgery. Remember Forgeo.**  
> Deep learning-powered real-vs-fake document detection built with PyTorch + Streamlit.

---

## 🌟 Features

- 📎 Upload scanned documents (JPG, JPEG, PNG)
- 🧠 EfficientNet-B0 based binary classification
- 📊 Real-time prediction with confidence score & chart
- 💡 Clean, animated UI with custom themes
- ⚡ Fast, local or cloud deployment (Streamlit)
- 🎨 Light, Dark, and Neon modes (easy to extend)
- 📦 Downloadable analysis report

---

## 🛠️ Tech Stack

| Layer      | Tech                        |
|------------|-----------------------------|
| Model      | EfficientNet-B0 (PyTorch)   |
| Interface  | Streamlit                   |
| Dataset    | DocXPand-25k                |
| Deployment | Streamlit / Local           |
| Animations | CSS Keyframes & Transitions |
| Themes     | Custom Streamlit Themes     |

---

## 📦 Installation

**Clone the Repo**
```sh
git clone https://github.com/yourusername/forgeo.git
cd forgeo
```

** If wan to run from Start then and Install Requirements**
```sh
python prepare_docxpand_dataset.py
Run cells in model_training.ipynb
pip install -r requirements.txt
```

**Run the App**
```sh
streamlit run app.py
```

---

## 🚀 Usage

1. **Upload** a document image (JPG, JPEG, PNG).
2. **Click** "Analyze Document".
3. **View** authenticity result, confidence, and analysis.
4. **Download** a detailed report if needed.

---

## 🎨 Themes & Animations

- **Themes:** Easily switch between Light, Dark, and Neon (add more in CSS).
- **Animations:** Smooth fade-in, slide, zoom, and loading spinner for a modern feel.

---

## 📂 Folder Structure

```
forgeo/
├── app.py               # Streamlit frontend
├── best_model.pth       # Trained EfficientNet model
├── requirements.txt     # Python dependencies
├── .gitignore           # Ignored files
└── README.md            # This file!
```

---

## 👥 Contributors

- [Sagnik Ghosh](https://github.com/sagnik7081) 
- [Sadashray Rastogi](https://github.com/Sadashrayr)


## 🙏 Acknowledgements

- [PyTorch](https://pytorch.org/)
- [Streamlit](https://streamlit.io/)
- [Plotly](https://plotly.com/python/)
- [DocXPand-25k Dataset](#)

---

**Upload a document → Get result → Trust the pixels


