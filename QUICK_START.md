# 🚀 Quick Start Guide - Misinformation Detection System

## Prerequisites
- Python 3.8+
- pip installed

## Installation (5 minutes)

### Step 1: Install Dependencies
```bash
cd d:\pro2
pip install flask flask-cors transformers torch scikit-learn networkx matplotlib plotly numpy pandas requests beautifulsoup4 scipy
```

**First-time note:** BERT models (~440MB) will download automatically on first run.

### Step 2: Start the Server
```bash
python app.py
```

You should see:
```
===========================================================
🚀 Misinformation Detection & Propagation Analysis System
===========================================================

📊 Starting Flask server...
🌐 Open your browser to: http://localhost:5000

===========================================================
```

### Step 3: Open in Browser
Navigate to: **http://localhost:5000**

---

## Using the Application

### **Option 1: Test with Sample Articles**
1. Click **"Fake News Sample"** or **"Real News Sample"** button
2. Click **"Analyze Information"**
3. Wait 6-7 seconds for analysis
4. View comprehensive results!

### **Option 2: Paste Your Own Text**
1. Select **"Text Input"** tab
2. Paste news article (minimum 20 characters)
3. Click **"Analyze Information"**

### **Option 3: Analyze from URL**
1. Select **"URL Input"** tab
2. Enter news article URL (e.g., from news websites)
3. Click **"Analyze Information"**

---

## What You'll See

### **Detection Results**
- ✅ Large verdict banner: **LIKELY MISINFORMATION** or **LIKELY AUTHENTIC**
- ✅ Confidence percentage
- ✅ BERT model prediction
- ✅ SVM baseline prediction

### **Propagation Analysis**
- ✅ Spread severity (High/Moderate)
- ✅ R0 reproduction number
- ✅ Potential reach percentage
- ✅ Peak infection day

### **Visual Analysis (6 Charts)**
1. Fake vs Real distribution (pie chart)
2. BERT vs SVM accuracy comparison (bar chart)
3. Processing pipeline (line chart)
4. SIR propagation model (time series)
5. Network graph (interactive)
6. Top influencers (pie chart)

---

## Troubleshooting

### **Issue: Dependencies fail to install**
Try installing individually:
```bash
pip install flask
pip install transformers
pip install torch
pip install scikit-learn
```

### **Issue: BERT model download fails**
- Check internet connection
- Model downloads automatically on first prediction
- Requires ~440MB of disk space

### **Issue: Port 5000 already in use**
Change port in `app.py`:
```python
app.run(debug=True, host='0.0.0.0', port=5001)  # Change to 5001
```

### **Issue: Analysis takes too long**
- Normal time: 6-7 seconds (CPU)
- BERT inference is the slowest part (2-3s)
- To speed up: Use GPU with CUDA-enabled PyTorch

---

## Project Structure

```
d:\pro2\
├── app.py              ← Start here (Flask server)
├── templates/
│   └── index.html      ← Frontend UI
├── static/
│   └── style.css       ← Styling
├── models/             ← AI models
├── propagation/        ← SIR & Network analysis
├── visualization/      ← Charts generation
├── data/               ← Dataset
└── README.md           ← Full documentation
```

---

## For Viva/Demonstration

1. **Start server** before presenting
2. Use **sample buttons** for quick demo
3. Show **fake news analysis** first (more dramatic results)
4. Highlight **each visualization** and explain
5. Reference **README.md** for technical questions

**Pro tip:** Keep `README.md` open during viva - it has 20+ Q&A prepared!

---

## Key Features to Highlight

🧠 **BERT AI Model** - 110M parameters, 87% accuracy  
📊 **SIR Epidemiological Model** - Simulates real spread patterns  
🌐 **Network Analysis** - Identifies top influencers  
📈 **6 Visualizations** - Publication-ready charts  
💎 **Premium UI** - Glassmorphism design with animations  
📚 **Academic Documentation** - Research-grade README  

---

## Next Steps

1. ✅ **Test thoroughly** with different articles
2. ✅ **Read README.md** for viva preparation
3. ✅ **Practice demo** presentation
4. ✅ **Customize** with your name/details if needed
5. ✅ **Submit** with confidence!

---

## Support

- Main documentation: `README.md`
- Walkthrough: Check artifacts folder
- Code comments: All files are well-commented

---

**🎉 You're all set! Good luck with your presentation! 🎉**
