# ab-testing-user-behavior
A/B testing analysis to evaluate user behavior, conversion rates, and session duration using Python and statistical testing
Here's the updated `README.md` without the license section:

---

```markdown
# A/B Testing: User Behavior Analysis

This project performs an A/B testing analysis to determine which version of a website performs better in terms of user behavior, session duration, and conversions. The analysis uses Python for data processing, visualization, and statistical testing.

---

## 📊 Project Overview

A/B testing is a method to compare two versions of a product (Variant A and Variant B) and analyze which performs better on key metrics. This project involves:

- Reading and cleaning user session data
- Visualizing distributions and trends
- Running hypothesis tests (t-tests)
- Generating a business summary report

---

## 📁 Project Structure

```

ab-testing-user-behavior/
├── data/
│   └── ab\_test\_web\_data.csv          # Raw dataset
├── src/
│   └── analysis.py                   # Main analysis and report generation script
├── outputs/
│   ├── plots/                        # Generated visualizations
│   └── ab\_test\_report.txt           # Final business report
├── requirements.txt                 # Python dependencies
└── README.md                        # Project documentation

````

---

## 🔧 Installation

1. **Clone the repository**
```bash
git clone https://github.com/your-username/ab-testing-user-behavior.git
cd ab-testing-user-behavior
````

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Run the analysis**

```bash
python src/analysis.py
```

---

## 📈 Key Features

* 📂 Handles real-world A/B test dataset
* 📊 Visualizes key metrics: session duration, conversion rate
* 📉 Performs t-tests to assess statistical significance
* 📝 Automatically generates a business-friendly report
* 🖼 Saves visual outputs for presentations or dashboards

---

## 📦 Requirements

* Python 3.10+
* pandas
* matplotlib
* seaborn
* scipy
* numpy

Install with:

```bash
pip install pandas matplotlib seaborn scipy numpy
```

---

## 📃 Sample Output

* ✅ Statistical test results (p-values, means)
* 📈 Graphs showing differences in user behavior
* 🧾 Final business report with recommendations

---

## ✍️ Author

**Aniket Ingle**


