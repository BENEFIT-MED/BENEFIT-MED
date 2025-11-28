import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from scipy.stats import shapiro, f_oneway, kruskal, ttest_ind, mannwhitneyu, levene, pearsonr
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import scikit_posthocs as sp
import io
import csv
import json
from datetime import datetime

st.set_page_config(layout="wide")
st.markdown("""
    <style>
        .sticky-image {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            background-color: white;
            z-index: 800;
            border-bottom: 2px solid lightblue;
            padding: 0;
            margin: 0;
            text-align: center;
        }
        .sticky-image img {
            width: 90%;
            height: auto;
            display: block;
        }
        .main-content {
            padding-top: 120px;
        }
        .stat-test {
            background-color: #f0f8ff;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 15px;
            border-left: 4px solid #1e90ff;
        }
        .section-header {
            background: linear-gradient(90deg, #1e90ff, #87cefa);
            color: white;
            padding: 10px 15px;
            border-radius: 5px;
            margin: 10px 0;
        }
        .multi-group-warning {
            background-color: #fff3cd;
            border: 1px solid #ffeaa7;
            border-radius: 5px;
            padding: 10px;
            margin: 10px 0;
        }
        .posthoc-comparison {
            background-color: #e8f5e8;
            border: 1px solid #c8e6c9;
            border-radius: 5px;
            padding: 10px;
            margin: 5px 0;
        }
        .complex-comparison {
            background-color: #e3f2fd;
            border: 1px solid #90caf9;
            border-radius: 5px;
            padding: 10px;
            margin: 5px 0;
        }
    </style>
    <div class="sticky-image">
        <img src="https://i.imgur.com/F0dqfM8.png" />
    </div>
""", unsafe_allow_html=True)

st.markdown("<div class='main-content'>", unsafe_allow_html=True)

st.markdown("""
    <div style='text-align: center; margin-bottom: 20px;'>
        <h1 style='color: #2c3e50; margin-bottom: 10px;'>
            BENEFIT-MED DATA EXPLORER
        </h1>
        <h3 style='color: #7f8c8d; font-weight: normal;'>
            Interactive statistical analysis and visualization
        </h3>
    </div>
""", unsafe_allow_html=True)

def detect_separator(file):
    try:
        sample = file.read(2048).decode('utf-8', errors='ignore')
        file.seek(0)
        sniffer = csv.Sniffer()
        dialect = sniffer.sniff(sample, delimiters=[',', ';', '\t', '|', ':', ''])
        return dialect.delimiter
    except Exception:
        file.seek(0)
        first_line = file.readline().decode('utf-8', errors='ignore')
        file.seek(0)
        for sep in [',', ';', '\t', '|', ':', '']:
            if sep in first_line:
                return sep
        return ','

@st.cache_data
def generate_sample_data():
    np.random.seed(42)
    n = 300
    genotypes = ['Wild Type', 'Mutant A', 'Mutant B', 'Transgenic']
    tissues = ['Leaf', 'Root', 'Stem', 'Flower']
    treatments = ['Control', 'Drought', 'Salt', 'Cold', 'Heat']
    
    secondary_metabolites = [
        'Chlorogenic Acid', 'Rutin', 'Quercetin', 'Kaempferol', 
        'Caffeic Acid', 'Ferulic Acid', 'Sinapic Acid', 'Anthocyanins'
    ]
    
    df = pd.DataFrame({
        "Genotype": np.random.choice(genotypes, size=n, p=[0.4, 0.2, 0.2, 0.2]),
        "Tissue": np.random.choice(tissues, size=n, p=[0.4, 0.3, 0.2, 0.1]),
        "Treatment": np.random.choice(treatments, size=n, p=[0.3, 0.2, 0.2, 0.15, 0.15]),
        "Time_Point": np.random.choice(['0h', '6h', '12h', '24h', '48h', '72h'], size=n),
        "Growth_Condition": np.random.choice(['Field', 'Greenhouse', 'Growth Chamber'], size=n),
    })
    
    for met in secondary_metabolites:
        if met == 'Anthocyanins':
            base = np.random.normal(5, 2, n)
            stress_effect = np.where(df['Treatment'].isin(['Cold', 'Drought']), np.random.normal(10, 3, n), 0)
            genotype_effect = np.where(df['Genotype'] == 'Mutant A', np.random.normal(-3, 1, n), 0)
            df[met] = np.clip(base + stress_effect + genotype_effect, 0.5, 30).round(2)
        elif met in ['Chlorogenic Acid', 'Rutin']:
            base = np.random.normal(8, 3, n)
            tissue_effect = np.where(df['Tissue'].isin(['Leaf', 'Flower']), np.random.normal(5, 2, n), 0)
            df[met] = np.clip(base + tissue_effect, 0.5, 25).round(2)
        else:
            df[met] = np.clip(np.random.lognormal(1.5, 0.5, n), 0.1, 15).round(2)
    
    return df

def check_normality(data, alpha=0.05):
    if len(data) < 3:
        return False, "Not enough data points (n < 3)"
    stat, p = shapiro(data)
    if p > alpha:
        return True, f"Normally distributed (p={p:.4f})"
    else:
        return False, f"Not normally distributed (p={p:.4f})"

def check_variance_homogeneity(group_data, alpha=0.05):
    try:
        stat, p = levene(*group_data)
        if p > alpha:
            return True, f"Equal variances (p={p:.4f})"
        else:
            return False, f"Unequal variances (p={p:.4f})"
    except Exception:
        return False, "Could not perform variance test"

def perform_statistical_test(df, x_col, y_col, p_value_threshold=0.05):
    groups = df[x_col].dropna().unique()
    group_data = [df[df[x_col] == group][y_col].dropna() 
                 for group in groups if len(df[df[x_col] == group]) >= 3]

    if len(group_data) < 2:
        return None, "Not enough groups with sufficient data (need at least 2 groups with ≥3 samples each)"

    normality_results = [check_normality(g) for g in group_data]
    all_normal = all([result[0] for result in normality_results])
    variance_result = check_variance_homogeneity(group_data)
    equal_var = variance_result[0]

    if len(groups) > 2:
        if all_normal and equal_var:
            stat, p = f_oneway(*group_data)
            test_used = "ANOVA"
            posthoc_method = "Tukey HSD"
        else:
            stat, p = kruskal(*group_data)
            test_used = "Kruskal-Wallis"
            posthoc_method = "Dunn's test"
    else:
        g1, g2 = group_data[:2]
        if all_normal and equal_var:
            stat, p = ttest_ind(g1, g2, equal_var=True)
            test_used = "Independent t-test"
            posthoc_method = None
        else:
            stat, p = mannwhitneyu(g1, g2)
            test_used = "Mann-Whitney U"
            posthoc_method = None

    p_str = f"p < {p_value_threshold}" if p < p_value_threshold else f"p = {p:.4f}"
    interpretation = "Significant difference" if p < p_value_threshold else "No significant difference"

    return {
        "x_variable": x_col,
        "y_variable": y_col,
        "groups": groups.tolist(),
        "group_sizes": [len(g) for g in group_data],
        "normality": [result[1] for result in normality_results],
        "variance": variance_result[1],
        "all_normal": all_normal,
        "equal_var": equal_var,
        "test": test_used,
        "statistic": stat,
        "p_value": p,
        "p_str": p_str,
        "interpretation": interpretation,
        "posthoc_method": posthoc_method,
        "p_value_threshold": p_value_threshold
    }, None

def perform_posthoc_test(df, x_col, y_col, test_results, p_value_threshold=0.05):
    if test_results["posthoc_method"] is None:
        return None
    
    data = df[[x_col, y_col]].dropna()
    
    if test_results["posthoc_method"] == "Tukey HSD":
        tukey = pairwise_tukeyhsd(endog=data[y_col], groups=data[x_col], alpha=p_value_threshold)
        tukey_df = pd.DataFrame(data=tukey._results_table.data[1:], columns=tukey._results_table.data[0])
        
        sig_results = tukey_df[tukey_df['reject']].sort_values('meandiff', key=abs, ascending=False)
        non_sig_results = tukey_df[~tukey_df['reject']].sort_values('meandiff', key=abs, ascending=False)
        
        return {
            "method": "Tukey HSD",
            "all_results": tukey_df,
            "significant_results": sig_results,
            "non_significant_results": non_sig_results,
            "summary": tukey.summary(),
            "alpha": p_value_threshold
        }
    
    elif test_results["posthoc_method"] == "Dunn's test":
        dunn_results = sp.posthoc_dunn(data, val_col=y_col, group_col=x_col, p_adjust='holm')
        
        significant_pairs = []
        non_significant_pairs = []
        
        for i in range(len(dunn_results)):
            for j in range(i+1, len(dunn_results)):
                pair_data = {
                    'group1': dunn_results.index[i],
                    'group2': dunn_results.columns[j],
                    'p_value': dunn_results.iloc[i,j]
                }
                if dunn_results.iloc[i,j] < p_value_threshold:
                    significant_pairs.append(pair_data)
                else:
                    non_significant_pairs.append(pair_data)

        significant_pairs.sort(key=lambda x: x['p_value'])
        non_significant_pairs.sort(key=lambda x: x['p_value'])
        
        return {
            "method": "Dunn's test",
            "results": dunn_results,
            "significant_pairs": significant_pairs,
            "non_significant_pairs": non_significant_pairs,
            "alpha": p_value_threshold
        }
    
    return None

def create_interaction_variable(df, cat_vars):
    if len(cat_vars) == 0:
        return None, "No categorical variables selected"
    
    interaction_name = " × ".join(cat_vars)
    df[interaction_name] = df[cat_vars].apply(lambda x: ' | '.join(x.astype(str)), axis=1)
    
    n_groups = df[interaction_name].nunique()
    n_observations = len(df)
    
    return interaction_name, f"Created {n_groups} interaction groups from {len(cat_vars)} variables"

def perform_multiway_anova(df, cat_vars, y_var, p_value_threshold=0.05):
    try:
        formula = f"{y_var} ~ " + " * ".join(cat_vars)
        
        import statsmodels.api as sm
        from statsmodels.formula.api import ols
        
        model = ols(formula, data=df).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)

        results = {
            "method": "Multi-way ANOVA",
            "formula": formula,
            "anova_table": anova_table,
            "significant_effects": [],
            "interaction_effects": []
        }
        
        for effect in anova_table.index:
            p_val = anova_table.loc[effect, 'PR(>F)']
            if p_val < p_value_threshold:
                effect_type = "Interaction" if ':' in effect or '*' in effect else "Main effect"
                results["significant_effects"].append({
                    'effect': effect,
                    'F_value': anova_table.loc[effect, 'F'],
                    'p_value': p_val,
                    'type': effect_type
                })
        
        return results, None
        
    except Exception as e:
        return None, f"Error performing multi-way ANOVA: {str(e)}"

def create_interaction_plot(df, cat_vars, y_var, p_value_threshold=0.05):
    if len(cat_vars) < 2:
        return None
    
    if len(cat_vars) == 2:
        fig = px.box(df, x=cat_vars[0], y=y_var, color=cat_vars[1],
                    title=f"Interaction Plot: {y_var} by {cat_vars[0]} and {cat_vars[1]}")
    else:
        fig = px.box(df, x=cat_vars[0], y=y_var, color=cat_vars[1],
                    facet_col=cat_vars[2] if len(cat_vars) > 2 else None,
                    title=f"Interaction Plot: {y_var} by multiple factors")
    return fig

def create_posthoc_visualization(df, x_col, y_col, test_results, posthoc_results):
    if posthoc_results is None:
        return None
    fig = px.box(df, x=x_col, y=y_col, color=x_col, points="all",
                title=f"{y_col} by {x_col}<br>{test_results['test']}: {test_results['p_str']}")
    
    if posthoc_results["method"] == "Tukey HSD" and len(posthoc_results["significant_results"]) > 0:
        annotations = []
        y_max = df[y_col].max()
        y_range = df[y_col].max() - df[y_col].min()
        y_step = y_range * 0.08
        
        sig_results = posthoc_results["significant_results"]
        
        for i, (_, row) in enumerate(sig_results.iterrows()):
            group1 = row['group1']
            group2 = row['group2']
            p_value = row['p-adj']
            meandiff = row['meandiff']
            
            direction = ">" if meandiff > 0 else "<"
            if meandiff > 0:
                comparison_text = f"{group1} > {group2}"
            else:
                comparison_text = f"{group2} > {group1}"
            
            annotations.append(dict(
                xref='paper', yref='y',
                x=0.02, y=y_max + (i+1)*y_step,
                xanchor='left', yanchor='bottom',
                text=f"{comparison_text} (p={p_value:.4f}, Δ={meandiff:.3f})",
                showarrow=False,
                font=dict(size=11, color='red'),
                bgcolor='rgba(255, 255, 255, 0.8)',
                bordercolor='red',
                borderwidth=1,
                borderpad=4
            ))
        
        fig.update_layout(annotations=annotations, height=500)
    
    elif posthoc_results["method"] == "Dunn's test" and len(posthoc_results["significant_pairs"]) > 0:
        annotations = []
        y_max = df[y_col].max()
        y_range = df[y_col].max() - df[y_col].min()
        y_step = y_range * 0.08
        
        for i, pair in enumerate(posthoc_results["significant_pairs"]):
            annotations.append(dict(
                xref='paper', yref='y',
                x=0.02, y=y_max + (i+1)*y_step,
                xanchor='left', yanchor='bottom',
                text=f"{pair['group1']} vs {pair['group2']} (p={pair['p_value']:.4f})",
                showarrow=False,
                font=dict(size=11, color='red'),
                bgcolor='rgba(255, 255, 255, 0.8)',
                bordercolor='red',
                borderwidth=1,
                borderpad=4
            ))
        
        fig.update_layout(annotations=annotations, height=500)
    
    return fig

def display_posthoc_results(posthoc_results, p_value_threshold):
    st.markdown(f"#### 🔍 Post-hoc Analysis ({posthoc_results['method']})")
    
    if posthoc_results["method"] == "Tukey HSD":
        total_comparisons = len(posthoc_results["all_results"])
        sig_comparisons = len(posthoc_results["significant_results"])
        st.info(f"**Total comparisons:** {total_comparisons} | **Significant comparisons:** {sig_comparisons} (p < {p_value_threshold})")
        
        if len(posthoc_results["significant_results"]) > 0:
            st.markdown("##### ✅ Significant Comparisons")
            for _, row in posthoc_results["significant_results"].iterrows():
                direction = ">" if row['meandiff'] > 0 else "<"
                if row['meandiff'] > 0:
                    comparison_text = f"{row['group1']} > {row['group2']}"
                else:
                    comparison_text = f"{row['group2']} > {row['group1']}"
                
                st.markdown(f"""
                <div class="posthoc-comparison">
                    <strong>{comparison_text}</strong><br>
                    Mean difference: {row['meandiff']:.4f} | 
                    p-value: {row['p-adj']:.4f} | 
                    CI: [{row['lower']:.4f}, {row['upper']:.4f}]
                </div>
                """, unsafe_allow_html=True)
        
        with st.expander("Show non-significant comparisons"):
            if len(posthoc_results["non_significant_results"]) > 0:
                for _, row in posthoc_results["non_significant_results"].iterrows():
                    st.write(f"{row['group1']} vs {row['group2']}: p = {row['p-adj']:.4f}")
            else:
                st.info("All comparisons are statistically significant")
    
    elif posthoc_results["method"] == "Dunn's test":
        total_comparisons = len(posthoc_results["significant_pairs"]) + len(posthoc_results["non_significant_pairs"])
        sig_comparisons = len(posthoc_results["significant_pairs"])
        st.info(f"**Total comparisons:** {total_comparisons} | **Significant comparisons:** {sig_comparisons} (p < {p_value_threshold})")
        
        if len(posthoc_results["significant_pairs"]) > 0:
            st.markdown("##### ✅ Significant Comparisons")
            for pair in posthoc_results["significant_pairs"]:
                st.markdown(f"""
                <div class="posthoc-comparison">
                    <strong>{pair['group1']} vs {pair['group2']}</strong><br>
                    p-value: {pair['p_value']:.4f}
                </div>
                """, unsafe_allow_html=True)
        
        with st.expander("Show non-significant comparisons"):
            if len(posthoc_results["non_significant_pairs"]) > 0:
                for pair in posthoc_results["non_significant_pairs"]:
                    st.write(f"{pair['group1']} vs {pair['group2']}: p = {pair['p_value']:.4f}")
            else:
                st.info("All comparisons are statistically significant")

with st.sidebar:
    st.image("https://i.imgur.com/hoAtwrV.jpeg", use_container_width=True)
    
    st.markdown('<div class="section-header">📂 Data Upload</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload File", type=["csv", "xlsx", "json"], 
                                   help="Upload your dataset in CSV, Excel, or JSON format",
                                   label_visibility="collapsed")
    
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                sep = detect_separator(uploaded_file)
                df = pd.read_csv(uploaded_file, sep=sep)
            elif uploaded_file.name.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(uploaded_file)
            elif uploaded_file.name.endswith('.json'):
                df = pd.read_json(uploaded_file)
            else:
                raise ValueError("Unsupported file format")
            
            for col in df.columns:
                if df[col].dtype == 'object':
                    try:
                        df[col] = pd.to_datetime(df[col])
                    except (ValueError, TypeError):
                        continue
            
            st.session_state["df"] = df
            st.success(f"✅ {uploaded_file.name} loaded successfully")
            
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
            df = generate_sample_data()
            st.session_state["df"] = df
            st.info("💡 Using sample data instead.")
    else:
        df = generate_sample_data()
        st.session_state["df"] = df
        st.info("💡 Using a sample dataset. Upload your own data to analyze.")

    df = st.session_state.get("df", generate_sample_data())
    cat_vars = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    num_vars = df.select_dtypes(include=["number"]).columns.tolist()
    
    st.markdown('<div class="section-header">🔍 Data Filters</div>', unsafe_allow_html=True)
    with st.expander("Categorical Filters", expanded=False):
        for col in cat_vars:
            options = st.multiselect(
                f"Filter {col}",
                df[col].dropna().unique(),
                default=df[col].dropna().unique(),
                key=f"filter_{col}"
            )
            if options:
                df = df[df[col].isin(options)]

    st.markdown('<div class="section-header">📊 Analysis Settings</div>', unsafe_allow_html=True)
    
    p_value_threshold = st.select_slider(
        "P-value threshold for statistical significance",
        options=[0.001, 0.01, 0.05, 0.1],
        value=0.05,
        help="Threshold for considering results statistically significant"
    )
    st.info(f"Using p-value threshold: **{p_value_threshold}**")
    
    analysis_type = st.radio(
        "Analysis Type",
        ["Descriptive", "Comparative", "Correlational"],
        index=0,
        help="Select the type of analysis to perform"
    )

    if analysis_type == "Descriptive":
        selected_num_vars = st.multiselect(
            "Numerical Variables for Analysis",
            options=num_vars,
            default=num_vars[:min(3, len(num_vars))],
            help="Select numerical variables for descriptive analysis"
        )
        selected_cat_vars = st.multiselect(
            "Categorical Variables for Analysis",
            options=cat_vars,
            default=cat_vars[:min(2, len(cat_vars))],
            help="Select categorical variables for frequency analysis"
        )
        
    elif analysis_type == "Comparative":
        x_vars = st.multiselect(
            "Grouping Variables (Categorical)",
            options=cat_vars,
            help="Select one or more categorical variables for comparison"
        )
        
        y_var = st.selectbox(
            "Outcome Variable (Numerical)",
            options=num_vars,
            help="Select numerical variable to compare"
        )
        
        if len(x_vars) > 1:
            analysis_method = st.radio(
                "Analysis Method",
                ["Interaction Variable", "Multi-way ANOVA"],
                help="Choose how to analyze multiple categorical variables"
            )
        else:
            analysis_method = "Single Variable"
        graph_type = st.selectbox(
            "Graph Type",
            ["Box Plot", "Violin Plot", "Strip Plot", "Bar Plot", "Interaction Plot"],
            help="Select the type of visualization to display"
        )
        
        show_posthoc = st.checkbox(
            "Show post-hoc tests",
            value=True,
            help="Show pairwise comparisons if overall test is significant"
        )
        
        if len(x_vars) > 2:
            st.warning("⚠️ Analysis with more than 2 categorical variables may produce complex results. Consider simplifying your analysis.")
        
    elif analysis_type == "Correlational":
        col1, col2 = st.columns(2)
        with col1:
            x_var_corr = st.selectbox("X Variable (Numerical)", options=num_vars)
        with col2:
            y_var_corr = st.selectbox("Y Variable (Numerical)", options=[v for v in num_vars if v != x_var_corr])
        
        graph_type_corr = st.selectbox("Graph Type", ["Scatter Plot", "Line Plot", "Hexbin Plot", "Density Contour"])
        color_var_corr = st.selectbox("Color by (Optional)", options=["None"] + cat_vars)
        color_var_corr = None if color_var_corr == "None" else color_var_corr
        trendline = st.checkbox("Show trendline", value=True)
        show_correlation_matrix = st.checkbox("Show correlation matrix", value=False)

st.markdown('<div class="section-header">🔍 Data Preview</div>', unsafe_allow_html=True)
with st.expander("View Dataset", expanded=False):
    display_rows = st.slider("Number of rows to display", 10, 500, 100)
    st.dataframe(df.head(display_rows), height=300, use_container_width=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Rows", len(df))
    with col2:
        st.metric("Total Columns", len(df.columns))
    with col3:
        st.metric("Missing Values", df.isnull().sum().sum())
    with col4:
        st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

st.markdown('<div class="section-header">📈 Statistical Analysis</div>', unsafe_allow_html=True)

if analysis_type == "Descriptive":
    st.markdown("### Descriptive Analysis")
    
    if not selected_num_vars:
        st.warning("Please select at least one numerical variable for descriptive analysis.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            stats_level = st.selectbox("Statistics Level", ["Basic", "Detailed", "Complete"])
            show_outliers = st.checkbox("Show outliers detection", value=True)
        with col2:
            include_plots = st.checkbox("Include distribution plots", value=True)
            group_by_vars = st.multiselect(
                "Group by (Optional - multiple selection)",
                options=cat_vars,
                help="Select one or more categorical variables to group the analysis"
            )
        
        if len(group_by_vars) > 2:
            st.markdown("""
            <div class="multi-group-warning">
                ⚠️ <strong>Note:</strong> Using more than 2 grouping variables may result in very specific subgroups 
                with small sample sizes. Consider limiting to 1-2 variables for clearer insights.
            </div>
            """, unsafe_allow_html=True)

        st.markdown("#### 📊 Descriptive Statistics")
        
        if group_by_vars:
            st.markdown(f"**Grouped by:** {', '.join(group_by_vars)}")
            stats_list = []
            
            for num_var in selected_num_vars:
                group_stats = df.groupby(group_by_vars)[num_var].agg([
                    'count', 'mean', 'std', 'min', 'max', 'median'
                ]).round(3)
                
                group_stats = group_stats.reset_index()
                group_stats['variable'] = num_var
                
                cols = ['variable'] + group_by_vars + ['count', 'mean', 'std', 'min', 'max', 'median']
                group_stats = group_stats[cols]
                
                stats_list.append(group_stats)
            
            if stats_list:
                stats_df = pd.concat(stats_list, ignore_index=True)
                st.dataframe(stats_df, use_container_width=True)
                
                group_combinations = df[group_by_vars].drop_duplicates()
                st.caption(f"**Number of unique groups:** {len(group_combinations)}")
                
                group_sizes = df.groupby(group_by_vars).size().reset_index(name='count')
                st.caption(f"**Group sizes:**")
                st.dataframe(group_sizes, use_container_width=True, height=200)
        else:
            stats_data = []
            for var in selected_num_vars:
                desc_stats = df[var].describe()
                stats_row = {
                    'variable': var,
                    'count': desc_stats['count'],
                    'mean': desc_stats['mean'],
                    'std': desc_stats['std'],
                    'min': desc_stats['min'],
                    'max': desc_stats['max'],
                    'median': desc_stats['50%'],
                    'missing': df[var].isnull().sum()
                }
                
                if stats_level in ["Detailed", "Complete"]:
                    stats_row.update({
                        'Q1': df[var].quantile(0.25),
                        'Q3': df[var].quantile(0.75),
                        'IQR': df[var].quantile(0.75) - df[var].quantile(0.25),
                        'skewness': df[var].skew(),
                        'kurtosis': df[var].kurtosis()
                    })
                
                if stats_level == "Complete":
                    stats_row.update({
                        'variance': df[var].var(),
                        'range': df[var].max() - df[var].min(),
                        'CV': (df[var].std() / df[var].mean()) if df[var].mean() != 0 else np.nan
                    })
                
                stats_data.append(stats_row)
            
            stats_df = pd.DataFrame(stats_data)
            st.dataframe(stats_df.round(3), use_container_width=True)

        if show_outliers and not group_by_vars:
            st.markdown("#### 🚨 Outliers Detection (IQR Method)")
            outliers_data = []
            for var in selected_num_vars:
                Q1 = df[var].quantile(0.25)
                Q3 = df[var].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = df[(df[var] < lower_bound) | (df[var] > upper_bound)]
                
                outliers_data.append({
                    'Variable': var,
                    'Lower Bound': round(lower_bound, 3),
                    'Upper Bound': round(upper_bound, 3),
                    'Outliers Count': len(outliers),
                    'Outliers %': round(len(outliers) / len(df) * 100, 2)
                })
            
            outliers_df = pd.DataFrame(outliers_data)
            st.dataframe(outliers_df, use_container_width=True)

        if include_plots:
            st.markdown("#### 📈 Distribution Visualizations")
            
            for var in selected_num_vars:
                st.markdown(f"##### {var}")
                
                if group_by_vars:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        color_var_plot = group_by_vars[0] if group_by_vars else None
                        fig_hist = px.histogram(
                            df, x=var, 
                            title=f"Distribution of {var} by {color_var_plot}",
                            marginal="box", 
                            nbins=30, 
                            color=color_var_plot, 
                            opacity=0.7
                        )
                        st.plotly_chart(fig_hist, use_container_width=True)
                    
                    with col2:
                        if len(group_by_vars) == 1:                            
                            fig_box = px.box(
                                df, x=group_by_vars[0], y=var,
                                title=f"{var} by {group_by_vars[0]}",
                                color=group_by_vars[0]
                            )
                        elif len(group_by_vars) >= 2:                            
                            fig_box = px.box(
                                df, x=group_by_vars[0], y=var,
                                color=group_by_vars[1],
                                title=f"{var} by {group_by_vars[0]} and {group_by_vars[1]}"
                            )
                        st.plotly_chart(fig_box, use_container_width=True)                    
                   
                    if len(group_by_vars) >= 2:
                        st.markdown(f"###### Interactive Group Exploration for {var}")
                                              
                        facet_col = st.selectbox(
                            f"Facet variable for {var}",
                            options=group_by_vars,
                            key=f"facet_{var}"
                        )
                        
                        color_var_facet = st.selectbox(
                            f"Color variable for {var}",
                            options=[v for v in group_by_vars if v != facet_col],
                            key=f"color_facet_{var}"
                        )
                        
                        fig_facet = px.box(
                            df, x=color_var_facet, y=var,
                            color=color_var_facet,
                            facet_col=facet_col,
                            title=f"{var} by {facet_col} and {color_var_facet}",
                            height=400
                        )
                        st.plotly_chart(fig_facet, use_container_width=True)
                        
                else:                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig_hist = px.histogram(
                            df, x=var, 
                            title=f"Distribution of {var}",
                            marginal="box",
                            nbins=30,
                            opacity=0.7
                        )
                        st.plotly_chart(fig_hist, use_container_width=True)
                    
                    with col2:
                        fig_box = px.box(df, y=var, title=f"Box Plot of {var}")
                        st.plotly_chart(fig_box, use_container_width=True)                
                
                if not group_by_vars:
                    data_clean = df[var].dropna()
                    if len(data_clean) >= 3:
                        normal, norm_msg = check_normality(data_clean)
                        st.caption(f"**Normality test:** {norm_msg}")
       
        if selected_cat_vars:
            st.markdown("#### 📋 Categorical Variables Summary")
            for cat_var in selected_cat_vars:
                if group_by_vars:                    
                    contingency_data = []
                    for group_var in group_by_vars:
                        cross_tab = pd.crosstab(df[cat_var], df[group_var], margins=True)
                        st.markdown(f"**Cross-tabulation: {cat_var} × {group_var}**")
                        st.dataframe(cross_tab, use_container_width=True)                        
                        
                        cross_tab_pct = pd.crosstab(df[cat_var], df[group_var], normalize='columns') * 100
                        st.markdown(f"**Percentages by column: {cat_var} × {group_var}**")
                        st.dataframe(cross_tab_pct.round(2), use_container_width=True)                
               
                freq_table = df[cat_var].value_counts().reset_index()
                freq_table.columns = ['Category', 'Count']
                freq_table['Percentage'] = (freq_table['Count'] / len(df) * 100).round(2)
                
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.markdown(f"**Overall distribution of {cat_var}**")
                    st.dataframe(freq_table, use_container_width=True)
                with col2:
                    fig_bar = px.bar(freq_table, x='Category', y='Count', 
                                   title=f"Distribution of {cat_var}")
                    st.plotly_chart(fig_bar, use_container_width=True)

elif analysis_type == "Comparative":
    if not x_vars:
        st.warning("Please select at least one categorical variable for comparative analysis.")
    else:
        st.markdown(f"### Comparative Analysis: {y_var} by {', '.join(x_vars)}")        
        
        if len(x_vars) == 1:           
            x_var = x_vars[0]
            test_results, error = perform_statistical_test(df, x_var, y_var, p_value_threshold)
            
            if error:
                st.error(error)
            else:               
                st.markdown(f"""
                <div class="stat-test">
                    <h4>📊 Statistical Test Results</h4>
                    <p><strong>Test used:</strong> {test_results["test"]}</p>
                    <p><strong>Test statistic:</strong> {test_results["statistic"]:.4f}</p>
                    <p><strong>P-value:</strong> {test_results["p_str"]}</p>
                    <p><strong>Interpretation:</strong> {test_results["interpretation"]}</p>
                    <p><strong>Significance threshold:</strong> {p_value_threshold}</p>
                    <p><strong>Number of groups:</strong> {len(test_results["groups"])}</p>
                    <p><strong>Group sizes:</strong> {test_results["group_sizes"]}</p>
                </div>
                """, unsafe_allow_html=True)                
               
                if show_posthoc and test_results["posthoc_method"] is not None:
                    posthoc_results = perform_posthoc_test(df, x_var, y_var, test_results, p_value_threshold)
                    
                    if posthoc_results:                        
                        display_posthoc_results(posthoc_results, p_value_threshold)                        
                       
                        st.markdown("##### 📊 Post-hoc Visualizations")
                        posthoc_fig = create_posthoc_visualization(df, x_var, y_var, test_results, posthoc_results)
                        if posthoc_fig:
                            st.plotly_chart(posthoc_fig, use_container_width=True)
        
        else:           
            if analysis_method == "Interaction Variable":                
                interaction_var, message = create_interaction_variable(df, x_vars)
                st.info(f"**Interaction variable created:** {message}")
                
                if interaction_var:                   
                    test_results, error = perform_statistical_test(df, interaction_var, y_var, p_value_threshold)
                    
                    if error:
                        st.error(error)
                    else:                        
                        st.markdown(f"""
                        <div class="stat-test">
                            <h4>📊 Statistical Test Results (Interaction)</h4>
                            <p><strong>Interaction variable:</strong> {interaction_var}</p>
                            <p><strong>Test used:</strong> {test_results["test"]}</p>
                            <p><strong>Test statistic:</strong> {test_results["statistic"]:.4f}</p>
                            <p><strong>P-value:</strong> {test_results["p_str"]}</p>
                            <p><strong>Interpretation:</strong> {test_results["interpretation"]}</p>
                            <p><strong>Number of interaction groups:</strong> {len(test_results["groups"])}</p>
                        </div>
                        """, unsafe_allow_html=True)                        
                        
                        if show_posthoc and test_results["posthoc_method"] is not None:
                            posthoc_results = perform_posthoc_test(df, interaction_var, y_var, test_results, p_value_threshold)
                            
                            if posthoc_results:                                
                                display_posthoc_results(posthoc_results, p_value_threshold)                                
                                
                                st.markdown("##### 📊 Interaction Visualizations")                                
                                
                                interaction_fig = create_interaction_plot(df, x_vars, y_var, p_value_threshold)
                                if interaction_fig:
                                    st.plotly_chart(interaction_fig, use_container_width=True)                                
                                
                                posthoc_fig = create_posthoc_visualization(df, interaction_var, y_var, test_results, posthoc_results)
                                if posthoc_fig:
                                    st.plotly_chart(posthoc_fig, use_container_width=True)
            
            elif analysis_method == "Multi-way ANOVA":               
                anova_results, error = perform_multiway_anova(df, x_vars, y_var, p_value_threshold)
                
                if error:
                    st.error(error)
                else:                    
                    st.markdown(f"""
                    <div class="stat-test">
                        <h4>📊 Multi-way ANOVA Results</h4>
                        <p><strong>Formula:</strong> {anova_results['formula']}</p>
                        <p><strong>Significant effects found:</strong> {len(anova_results['significant_effects'])}</p>
                    </div>
                    """, unsafe_allow_html=True)                    
                    
                    st.markdown("##### ANOVA Table")
                    st.dataframe(anova_results['anova_table'].round(4), use_container_width=True)                    
                    
                    if anova_results['significant_effects']:
                        st.markdown("##### ✅ Significant Effects")
                        for effect in anova_results['significant_effects']:
                            st.markdown(f"""
                            <div class="complex-comparison">
                                <strong>{effect['effect']}</strong> ({effect['type']})<br>
                                F-value: {effect['F_value']:.4f} | p-value: {effect['p_value']:.4f}
                            </div>
                            """, unsafe_allow_html=True)                    
                    
                    interaction_fig = create_interaction_plot(df, x_vars, y_var, p_value_threshold)
                    if interaction_fig:
                        st.plotly_chart(interaction_fig, use_container_width=True)        
        
        st.markdown("##### 📈 Main Visualization")
        try:
            fig = None

            if graph_type == "Box Plot":
                if len(x_vars) == 1:
                    fig = px.box(df, x=x_vars[0], y=y_var, color=x_vars[0], points="all",
                                title=f"{y_var} by {x_vars[0]}")
                else:                   
                    fig = px.box(df, x=x_vars[0], y=y_var, color=x_vars[1] if len(x_vars) > 1 else None,
                                title=f"{y_var} by {x_vars[0]} and {x_vars[1] if len(x_vars) > 1 else ''}")
            
            elif graph_type == "Violin Plot":
                if len(x_vars) == 1:
                    fig = px.violin(df, x=x_vars[0], y=y_var, color=x_vars[0], box=True,
                                   title=f"{y_var} by {x_vars[0]}")
                else:
                    fig = px.violin(df, x=x_vars[0], y=y_var, color=x_vars[1] if len(x_vars) > 1 else None, box=True,
                                   title=f"{y_var} by {x_vars[0]} and {x_vars[1] if len(x_vars) > 1 else ''}")
            
            elif graph_type == "Strip Plot":
                if len(x_vars) == 1:
                    fig = px.strip(df, x=x_vars[0], y=y_var, color=x_vars[0],
                                title=f"{y_var} by {x_vars[0]}")
                else:
                    fig = px.strip(df, x=x_vars[0], y=y_var, color=x_vars[1] if len(x_vars) > 1 else None,
                                title=f"{y_var} by {x_vars[0]} and {x_vars[1] if len(x_vars) > 1 else ''}")
            
            elif graph_type == "Bar Plot":
                if len(x_vars) == 1:                
                    df_agg = df.groupby(x_vars[0])[y_var].agg(['mean', 'std', 'count']).reset_index()
                    df_agg['se'] = df_agg['std'] / np.sqrt(df_agg['count'])
                    fig = px.bar(df_agg, x=x_vars[0], y='mean',
                                error_y='se', 
                                title=f"{y_var} by {x_vars[0]}",
                                labels={'mean': f'Mean {y_var}'},
                                hover_data={'mean': ':.2f', 'se': ':.2f', 'std': ':.2f', 'count': True})
                else:                    
                    df_agg = df.groupby(x_vars[:2])[y_var].mean().reset_index()
                    fig = px.bar(df_agg, x=x_vars[0], y=y_var, color=x_vars[1],
                                title=f"{y_var} by {x_vars[0]} and {x_vars[1]}",
                                barmode='group')
            
            elif graph_type == "Interaction Plot":
                fig = create_interaction_plot(df, x_vars, y_var, p_value_threshold)
                if not fig:
                    st.warning("Interaction plot requires at least 2 categorical variables.")                    
                    fig = px.box(df, x=x_vars[0], y=y_var, title=f"{y_var} by {x_vars[0]}")
                    
                    if fig is not None:
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.error(f"Could not create visualization for graph type: {graph_type}")
                        
        except Exception as e:
            st.error(f"Could not create visualization: {str(e)}")            
            try:
                fig_fallback = px.scatter(df, x=x_vars[0] if x_vars else df.index, y=y_var, 
                                        title=f"Basic plot of {y_var}")
                st.plotly_chart(fig_fallback, use_container_width=True)
                st.info("Displaying basic scatter plot as fallback")
            except:
                st.error("Unable to create any visualization with the current data and settings.")

elif analysis_type == "Correlational":
    st.markdown(f"### Correlational Analysis: {y_var_corr} vs {x_var_corr}")    
    
    try:
        corr_coef, corr_p_value = pearsonr(df[x_var_corr].dropna(), df[y_var_corr].dropna())
        corr_significant = corr_p_value < p_value_threshold
        corr_p_str = f"p < {p_value_threshold}" if corr_p_value < p_value_threshold else f"p = {corr_p_value:.4f}"
        interpretation = "Significant correlation" if corr_significant else "No significant correlation"        
        
        abs_corr = abs(corr_coef)
        if abs_corr > 0.7:
            strength = "Strong"
        elif abs_corr > 0.3:
            strength = "Moderate"
        else:
            strength = "Weak"
        direction = "positive" if corr_coef > 0 else "negative"
        
    except Exception as e:
        st.error(f"Error calculating correlation: {str(e)}")
        corr_coef = np.nan
        interpretation = "Could not compute correlation"
        strength = "N/A"
        direction = "N/A"
        corr_p_str = "N/A"
   
    st.markdown(f"""
    <div class="stat-test">
        <h4>📊 Correlation Analysis</h4>
        <p><strong>Pearson correlation coefficient (r):</strong> {corr_coef:.4f}</p>
        <p><strong>P-value:</strong> {corr_p_str}</p>
        <p><strong>Interpretation:</strong> {interpretation}</p>
        <p><strong>Correlation strength:</strong> {strength} {direction} correlation</p>
        <p><strong>Significance threshold:</strong> {p_value_threshold}</p>
    </div>
    """, unsafe_allow_html=True)
    
    if show_correlation_matrix and len(num_vars) > 1:
        st.markdown("#### Correlation Matrix")
        corr_matrix = df[num_vars].corr()
        fig_corr = px.imshow(corr_matrix, text_auto=True, aspect="auto", 
                           color_continuous_scale="RdBu", title="Correlation Matrix")
        st.plotly_chart(fig_corr, use_container_width=True)
    
    title = f"{y_var_corr} vs {x_var_corr} (r = {corr_coef:.4f}, {corr_p_str})"
    
    try:
        if graph_type_corr == "Scatter Plot":
            fig = px.scatter(df, x=x_var_corr, y=y_var_corr, color=color_var_corr,
                           trendline="ols" if trendline else None, title=title)
        elif graph_type_corr == "Line Plot":
            fig = px.line(df, x=x_var_corr, y=y_var_corr, color=color_var_corr, title=title)
        elif graph_type_corr == "Hexbin Plot":
            fig = px.density_heatmap(df, x=x_var_corr, y=y_var_corr, title=title)
        elif graph_type_corr == "Density Contour":
            fig = px.density_contour(df, x=x_var_corr, y=y_var_corr, color=color_var_corr, title=title)
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Could not create visualization: {str(e)}")

with st.expander("💾 Export Data", expanded=False):
    st.write("### Export Options")
    export_format = st.radio("Export Format", ["CSV", "Excel", "JSON"], horizontal=True)
    export_filename = st.text_input("Filename", value=f"data_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    
    if st.button("Export Data"):
        try:
            buffer = io.BytesIO()
            if export_format == "CSV":
                df.to_csv(buffer, index=False)
                mime_type = "text/csv"
                file_ext = "csv"
            elif export_format == "Excel":
                with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                    df.to_excel(writer, index=False, sheet_name='Data')
                mime_type = "application/vnd.ms-excel"
                file_ext = "xlsx"
            elif export_format == "JSON":
                json_data = df.to_json(orient='records', indent=2)
                buffer.write(json_data.encode())
                mime_type = "application/json"
                file_ext = "json"
            
            buffer.seek(0)
            st.download_button(
                f"Download {export_format}",
                data=buffer,
                file_name=f"{export_filename}.{file_ext}",
                mime=mime_type
            )
            st.success("Data ready for download!")
        except Exception as e:
            st.error(f"Error exporting data: {str(e)}")

st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.9em; padding: 20px;'>
        <p>For educational and research purposes</p>
        <p>© 2025 From the BENEFIT-MED Project supported by the PRIMA funding Program</p>
        <p>Contact: 
            <a href='mailto:loic.rajjou@inrae.fr' style='color: #1e90ff;'>loic.rajjou@inrae.fr</a> |
            <a href='mailto:alma.balestrazzi@unipv.it' style='color: #1e90ff;'>alma.balestrazzi@unipv.it</a>
        </p>
    </div>
""", unsafe_allow_html=True)
