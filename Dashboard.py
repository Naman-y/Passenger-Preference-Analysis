import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

# Compatibility shim for older Plotly/xarray stacks on NumPy 2.x.
if not hasattr(np, "unicode_"):
    np.unicode_ = np.str_

import plotly.express as px

# ------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------
st.set_page_config(page_title="Airline Passenger Analysis", layout="wide")

st.title("✈ Airline Passenger Preference Dashboard")

# ------------------------------------------------
# LOAD DATA
# ------------------------------------------------
@st.cache_data
def load_data():
    data_path = Path(__file__).resolve().parent / "combined_dataset_cleaned.csv"
    df = pd.read_csv(data_path)

    # Convert Travel_Frequency to numeric for KPI
    mapping = {
        "0 flight": 0,
        "1 flight": 1,
        "2-3 flights": 2.5,
        "4-6 flights": 5,
        "7-10 flights": 8.5,
        "More than 10 flight": 12
    }

    df["Travel_Frequency_Numeric"] = df["Travel_Frequency"].map(mapping)

    return df

df = load_data()


@st.cache_data
def load_segmentation_data():
    data_path = Path(__file__).resolve().parent / "airline_segmentation_ready.csv"
    return pd.read_csv(data_path)


@st.cache_data
def load_model_ready_data():
    data_path = Path(__file__).resolve().parent / "airline_model_ready.csv"
    return pd.read_csv(data_path)


def run_kmeans(data, n_clusters, random_state=42, n_init=10, max_iter=100):
    X = np.asarray(data, dtype=float)
    rng = np.random.default_rng(random_state)

    best_labels = None
    best_centers = None
    best_inertia = np.inf

    for _ in range(n_init):
        initial_idx = rng.choice(len(X), size=n_clusters, replace=False)
        centers = X[initial_idx].copy()

        for _ in range(max_iter):
            distances = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
            labels = distances.argmin(axis=1)

            new_centers = centers.copy()
            for cluster_id in range(n_clusters):
                members = X[labels == cluster_id]
                if len(members) == 0:
                    new_centers[cluster_id] = X[rng.integers(0, len(X))]
                else:
                    new_centers[cluster_id] = members.mean(axis=0)

            if np.allclose(new_centers, centers):
                centers = new_centers
                break

            centers = new_centers

        final_distances = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
        final_labels = final_distances.argmin(axis=1)
        inertia = np.sum((X - centers[final_labels]) ** 2)

        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = final_labels
            best_centers = centers.copy()

    return best_labels, best_centers, best_inertia


def run_pca(data, n_components=2):
    X = np.asarray(data, dtype=float)
    X_centered = X - X.mean(axis=0)

    _, singular_values, vt = np.linalg.svd(X_centered, full_matrices=False)
    components = X_centered @ vt[:n_components].T

    explained_variance = (singular_values ** 2) / max(len(X) - 1, 1)
    total_variance = explained_variance.sum()

    if total_variance == 0:
        explained_ratio = np.zeros(n_components)
    else:
        explained_ratio = explained_variance[:n_components] / total_variance

    return components, explained_ratio


def prettify_feature_name(feature_name):
    label = str(feature_name).replace("_", " ").replace(";", "")
    replacements = {
        "Purpose of Travel ": "",
        "Influencing Factors ": "",
        "Reward Preference ": "",
        "Inflight Priority ": "",
        "Travel Frequency ": "",
        "Travel Class ": "",
        "Flight Preference ": "",
        "Booking Mode ": "",
        "Price Sensitivity ": "",
        "Loyalty Program ": "",
        "Schedule Preference ": "",
        "e.g": "eg",
    }

    for old, new in replacements.items():
        label = label.replace(old, new)

    return label.strip().title()


def compute_one_vs_rest_driver_scores(dataframe, target_col, airline_id, selected_features):
    airline_rows = dataframe[dataframe[target_col] == airline_id]
    other_rows = dataframe[dataframe[target_col] != airline_id]

    if airline_rows.empty or other_rows.empty:
        return pd.DataFrame(columns=["Feature", "Driver Score", "Airline Avg", "Others Avg", "Direction"])

    airline_mean = airline_rows[selected_features].mean()
    other_mean = other_rows[selected_features].mean()
    driver_diff = airline_mean - other_mean

    scores = pd.DataFrame({
        "Feature": selected_features,
        "Driver Score": driver_diff.abs().values,
        "Airline Avg": airline_mean.values,
        "Others Avg": other_mean.values,
        "Direction": np.where(driver_diff.values >= 0, "Higher for airline", "Lower for airline")
    })

    return scores.sort_values("Driver Score", ascending=False)

# ------------------------------------------------
# SIDEBAR NAVIGATION
# ------------------------------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Overview",
     "Demographics",
     "Travel Behavior",
     "Price & Loyalty",
     "Airline & Sentiment",
     "Customer Segmentation",
     "Airline-Specific Drivers"]
)

# =================================================
# 1 OVERVIEW PAGE
# =================================================
if page == "Overview":

    st.subheader("Executive Summary")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Passengers", len(df))

    col2.metric("Avg Travel Frequency",
                round(df["Travel_Frequency_Numeric"].mean(), 2))

    loyalty_status = df["Loyalty_Program"].astype(str).str.strip().str.lower()
    loyalty_rate = (loyalty_status == "yes").mean() * 100
    col3.metric("Loyalty Enrollment %",
                round(loyalty_rate, 2))

    st.info("Key Insight: Majority passengers fall in mid travel frequency range. Loyalty enrollment is moderate and concentrated in premium travelers.")

# =================================================
# 2 DEMOGRAPHICS PAGE
# =================================================
elif page == "Demographics":

    st.subheader("Passenger Demographics Analysis")

    col1, col2 = st.columns(2)

    # -----------------------------
    # Gender Distribution (Donut Chart)
    # -----------------------------
    with col1:
        fig1 = px.pie(
            df,
            names="Gender",
            hole=0.5,
            title="Gender Distribution"
        )
        st.plotly_chart(fig1, use_container_width=True)

    # -----------------------------
    # Age Distribution (Box Plot)
    # -----------------------------
    with col2:
        fig2 = px.box(
            df,
            y="Age",
            title="Age Distribution"
        )
        st.plotly_chart(fig2, use_container_width=True)

    # -----------------------------
    # Occupation Distribution (Clean Fix)
    # -----------------------------
    occupation_counts = df["Occupation"].value_counts().reset_index()
    occupation_counts.columns = ["Occupation", "Count"]

    fig3 = px.bar(
        occupation_counts,
        x="Occupation",
        y="Count",
        color="Occupation",
        title="Occupation Distribution"
    )

    st.plotly_chart(fig3, use_container_width=True)

    # -----------------------------
    # Travel Class by Gender (Stacked Bar)
    # -----------------------------
    fig4 = px.histogram(
        df,
        x="Travel_Class",
        color="Gender",
        barmode="group",
        title="Travel Class by Gender"
    )

    st.plotly_chart(fig4, use_container_width=True)

    # -----------------------------
    # Booking Mode by Age Group
    # -----------------------------
    booking_age = (
        df.groupby(["Age", "Booking_Mode"])
        .size()
        .reset_index(name="Count")
    )

    age_order = ["<18", "19-24", "25-34", "35-44", "45-60", "60+"]
    booking_age["Age"] = pd.Categorical(booking_age["Age"], categories=age_order, ordered=True)
    booking_age = booking_age.sort_values("Age")

    fig5 = px.bar(
        booking_age,
        x="Age",
        y="Count",
        color="Booking_Mode",
        barmode="group",
        title="Booking Mode by Age Group"
    )
    st.plotly_chart(fig5, use_container_width=True)

    st.info("""
    **Insights:**
    - Majority passengers belong to working professionals.
    - Gender split is relatively balanced, with males at 57%.
    - Age distribution is concentrated in the 25-34 active workforce band.
    - Travel class preferences vary across gender segments.
    - Younger passengers lean towards third-party booking platforms, while older segments prefer direct airline channels.
    """)

# =================================================
# 3. TRAVEL BEHAVIOR PAGE
# =================================================
elif page == "Travel Behavior":

    st.subheader("✈ Travel Behavior Analysis")

    col1, col2 = st.columns(2)

    # -----------------------------
    # Travel Frequency (Horizontal Bar)
    # -----------------------------
    with col1:
        tf_counts = df["Travel_Frequency"].value_counts().reset_index()
        tf_counts.columns = ["Travel_Frequency", "Count"]

        fig1 = px.bar(
            tf_counts.sort_values("Count"),
            x="Count",
            y="Travel_Frequency",
            orientation="h",
            color="Travel_Frequency",
            title="Travel Frequency Distribution"
        )
        st.plotly_chart(fig1, use_container_width=True)

    # -----------------------------
    # Travel Class (Donut Chart)
    # -----------------------------
    with col2:
        fig2 = px.pie(
            df,
            names="Travel_Class",
            hole=0.5,
            title="Travel Class Distribution"
        )
        st.plotly_chart(fig2, use_container_width=True)

    # -----------------------------
    # Purpose of Travel (Fixed Properly)
    # -----------------------------
    import re

    def clean_purpose(text):
        text = str(text).lower()
        parts = re.split(r',|/|;|-', text)
        return [p.strip() for p in parts]

    df["Purpose_Clean"] = df["Purpose_of_Travel"].apply(clean_purpose)
    df_exploded = df.explode("Purpose_Clean")

    # Define main categories only
    def map_category(x):
        if "business" in x:
            return "Business"
        elif "leisure" in x:
            return "Leisure"
        elif "family" in x:
            return "Family Visit"
        elif "education" in x:
            return "Education"
        elif "medical" in x:
            return "Medical"
        else:
            return "Others"

    df_exploded["Purpose_Grouped"] = df_exploded["Purpose_Clean"].apply(map_category)

    # Count only grouped categories
    purpose_counts = (
        df_exploded["Purpose_Grouped"]
        .value_counts()
        .reset_index()
    )

    purpose_counts.columns = ["Purpose", "Count"]

    # -----------------------------
    # CLEAN BAR CHART
    # -----------------------------
    fig = px.bar(
        purpose_counts.sort_values("Count"),
        x="Count",
        y="Purpose",
        orientation="h",
        color="Purpose",
        title="Purpose of Travel (Cleaned Categories)"
    )

    st.plotly_chart(fig, use_container_width=True)

    # -----------------------------
    # Travel Class vs Purpose (Stacked)
    # -----------------------------
    import re

    def clean_purpose(text):
        text = str(text).lower()
        # split by comma, slash, semicolon
        parts = re.split(r',|/|;', text)
        return [p.strip() for p in parts]

    df["Purpose_Clean"] = df["Purpose_of_Travel"].apply(clean_purpose)
    df_exploded = df.explode("Purpose_Clean")

    # Standardize main categories
    main_categories = {
        "business": "Business",
        "leisure": "Leisure",
        "family visit": "Family Visit",
        "education": "Education",
        "medical": "Medical"
    }

    def map_category(x):
        for key in main_categories:
            if key in x:
                return main_categories[key]
        return "Others"

    df_exploded["Purpose_Grouped"] = df_exploded["Purpose_Clean"].apply(map_category)

    # -----------------------------
    # COUNT ONLY MAIN GROUPS
    # -----------------------------
    purpose_counts = (
        df_exploded.groupby(["Purpose_Grouped", "Travel_Class"])
        .size()
        .reset_index(name="Count")
    )

    # -----------------------------
    # CLEAN GROUPED BAR CHART
    # -----------------------------
    fig4 = px.bar(
        purpose_counts,
        x="Purpose_Grouped",
        y="Count",
        color="Travel_Class",
        barmode="group",
        title="Purpose vs Travel Class"
    )

    st.plotly_chart(fig4, use_container_width=True)

    # -----------------------------
    # Influencing Factors Breakdown
    # -----------------------------
    import re as re_mod

    def parse_factors(text):
        text = str(text).lower()
        text = re_mod.sub(r"[\[\]']", "", text)
        parts = re_mod.split(r",|;", text)
        return [p.strip() for p in parts if p.strip()]

    df["Factors_Clean"] = df["Influencing_Factors"].apply(parse_factors)
    df_factors = df.explode("Factors_Clean")

    factor_counts = (
        df_factors["Factors_Clean"]
        .value_counts()
        .reset_index()
    )
    factor_counts.columns = ["Factor", "Count"]
    factor_counts["Factor"] = factor_counts["Factor"].str.replace("_", " ").str.title()

    fig_factors = px.bar(
        factor_counts.sort_values("Count"),
        x="Count",
        y="Factor",
        orientation="h",
        color="Factor",
        title="Top Influencing Factors for Airline Selection"
    )
    fig_factors.update_layout(showlegend=True)
    st.plotly_chart(fig_factors, use_container_width=True)

    # -----------------------------
    # Flight Preference by Travel Class
    # -----------------------------
    flight_pref = (
        df.groupby(["Flight_Preference", "Travel_Class"])
        .size()
        .reset_index(name="Count")
    )

    fig_fp = px.bar(
        flight_pref,
        x="Flight_Preference",
        y="Count",
        color="Travel_Class",
        barmode="group",
        title="Flight Preference by Travel Class"
    )
    st.plotly_chart(fig_fp, use_container_width=True)

    st.success("""
    **Business Insights:**
    - Majority passengers travel for business and leisure purposes.
    - Economy class dominates across most travel purposes.
    - Premium travel is concentrated among business travelers.
    - Ticket price and punctuality are the top factors influencing airline choice.
    - Direct flights are strongly preferred by premium class travelers.
    """)

# =================================================
# 4. PRICE & LOYALTY PAGE
# =================================================
elif page == "Price & Loyalty":

    st.subheader("  Price Sensitivity & Loyalty")

    col1, col2 = st.columns(2)

    with col1:
        fig1 = px.histogram(df,
                            x="Price_Sensitivity",
                            title="Price Sensitivity Distribution")
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        fig2 = px.histogram(df,
                            x="Price_Sensitivity",
                            color="Loyalty_Program",
                            barmode="group",
                            title="Price Sensitivity vs Loyalty")
        st.plotly_chart(fig2, use_container_width=True)

    fig3 = px.histogram(df,
                        x="Travel_Class",
                        color="Loyalty_Program",
                        barmode="group",
                        title="Travel Class vs Loyalty")
    st.plotly_chart(fig3, use_container_width=True)

    # -----------------------------
    # Age vs Price Sensitivity (NEW)
    # -----------------------------
    age_price = (
        df.groupby(["Age", "Price_Sensitivity"])
        .size()
        .reset_index(name="Count")
    )

    age_order = ["<18", "19-24", "25-34", "35-44", "45-60", "60+"]
    age_price["Age"] = pd.Categorical(age_price["Age"], categories=age_order, ordered=True)
    age_price = age_price.sort_values("Age")

    fig_age_price = px.bar(
        age_price,
        x="Age",
        y="Count",
        color="Price_Sensitivity",
        barmode="group",
        title="Age Group vs Price Sensitivity"
    )
    st.plotly_chart(fig_age_price, use_container_width=True)

    st.success("""
    **Business Insight:**
    - Price sensitivity distribution is skewed towards highly sensitive passengers.
    - Lower price sensitivity passengers show higher loyalty enrollment.
    - Business and Premium class customers are more likely to enroll in loyalty programs.
    - The 60+ age group shows the highest price sensitivity.
    - The 45-60 age group is the least price-sensitive, a sweet spot for premium upselling.
    """)

# =================================================
# 5️. AIRLINE & SENTIMENT PAGE
# =================================================
elif page == "Airline & Sentiment":

    st.subheader("Airline Preference Analysis")
    st.caption("Using the available columns from your combined dataset: `Airline_Last_Flown`, `Travel_Class`, and `Loyalty_Program`.")

    import re

    col1, col2 = st.columns(2)

    airline_source_col = None
    for candidate in ["Airline_List", "Airline_Last_Flown"]:
        if candidate in df.columns:
            airline_source_col = candidate
            break

    # -----------------------------
    # CLEAN AIRLINE DATA
    # -----------------------------
    def clean_airline(text):
        text = str(text).lower()
        text = re.sub(r'[\[\]\']', '', text)
        parts = re.split(r',|;', text)
        return [p.strip() for p in parts if p.strip()]

    if airline_source_col is None:
        st.error("No airline column was found in the dataset.")
        st.stop()

    df["Airline_Clean"] = df[airline_source_col].apply(clean_airline)
    df_airline = df.explode("Airline_Clean")

    def map_airline(x):
        if "indigo" in x:
            return "IndiGo"
        elif "air_india" in x or "air india" in x:
            return "Air India"
        elif "vistara" in x:
            return "Vistara"
        elif "spicejet" in x:
            return "SpiceJet"
        elif "akasa" in x:
            return "Akasa"
        else:
            return "Other"

    df_airline["Airline_Grouped"] = df_airline["Airline_Clean"].apply(map_airline)

    airline_counts = (
        df_airline["Airline_Grouped"]
        .value_counts()
        .reset_index()
    )

    airline_counts.columns = ["Airline", "Count"]

    # -----------------------------
    # Airline Preference Chart
    # -----------------------------
    with col1:
        fig1 = px.bar(
            airline_counts.sort_values("Count"),
            x="Count",
            y="Airline",
            orientation="h",
            color="Airline",
            title="Airline Preference "
        )
        fig1.update_layout(showlegend=False)
        st.plotly_chart(fig1, use_container_width=True)

   # -----------------------------
    # SECONDARY ANALYSIS
    # -----------------------------
    if "Sentiment_Label" in df.columns:

        with col2:
            fig2 = px.pie(
                df,
                names="Sentiment_Label",
                hole=0.5,
                title="Overall Sentiment"
            )
            st.plotly_chart(fig2, use_container_width=True)

        # Airline vs Sentiment
        df_sentiment = df_airline.copy()

        fig3 = px.histogram(
            df_sentiment,
            x="Airline_Grouped",
            color="Sentiment_Label",
            barmode="group",
            title="Airline vs Sentiment"
        )

        st.plotly_chart(fig3, use_container_width=True)

        st.info("**Insights:**\n\n• Market share is dominated by top airlines.\n\n• Positive sentiment varies by airline.\n\n• Operational consistency strongly affects brand perception.")
    else:
        with col2:
            travel_class_by_airline = (
                df_airline.groupby(["Airline_Grouped", "Travel_Class"])
                .size()
                .reset_index(name="Count")
            )

            fig2 = px.bar(
                travel_class_by_airline,
                x="Airline_Grouped",
                y="Count",
                color="Travel_Class",
                barmode="group",
                title="Airline Preference by Travel Class"
            )
            st.plotly_chart(fig2, use_container_width=True)

        if "Loyalty_Program" in df_airline.columns:
            loyalty_by_airline = (
                df_airline.groupby(["Airline_Grouped", "Loyalty_Program"])
                .size()
                .reset_index(name="Count")
            )

            fig3 = px.bar(
                loyalty_by_airline,
                x="Airline_Grouped",
                y="Count",
                color="Loyalty_Program",
                barmode="group",
                title="Airline vs Loyalty Program"
            )
            st.plotly_chart(fig3, use_container_width=True)

        st.info("**Insights:**\n\n• IndiGo and Air India appear most often in recent flown-airline records.\n\n• Travel class mix helps compare premium versus budget airline positioning.\n\n• Loyalty behavior still provides useful airline-level business insight without sentiment labels.")

# =================================================
# 6. Customer Segmentation
# =================================================
elif page == "Customer Segmentation":

    st.subheader("Interactive Customer Segmentation")

    st.markdown("""
    Explore airline customer segments using **K-Means clustering**. Adjust the inputs below to compare
    how traveler behavior, loyalty, price sensitivity, and service preferences shape each segment.
    """)

    df_seg = load_segmentation_data()

    feature_groups = {
        "Travel Purpose": [
            "business", "leisure", "family_visit", "education", "medical", "others"
        ],
        "Influencing Factors": [
            "Influencing_Factors_brand_reputation",
            "Influencing_Factors_free_food",
            "Influencing_Factors_in_flight_service_quality",
            "Influencing_Factors_loyalty_programs",
            "Influencing_Factors_punctuality",
            "Influencing_Factors_safety_rating",
            "Influencing_Factors_seat_comfort",
            "Influencing_Factors_ticket_price"
        ],
        "Reward Preferences": [
            "Reward_Preference_cashback_or_discount",
            "Reward_Preference_extra_baggage",
            "Reward_Preference_free_flights",
            "Reward_Preference_free_lounge_access",
            "Reward_Preference_free_seat_upgrades",
            "Reward_Preference_priority_check-in/boarding"
        ],
        "In-Flight Priorities": [
            "Inflight_Priority_amenities(e.g_charging_ports,_eye-mask,_blankets)",
            "Inflight_Priority_cabin_crew_behavior",
            "Inflight_Priority_extra_baggage_facility",
            "Inflight_Priority_extra_legroom",
            "Inflight_Priority_in-flight_entertainment",
            "Inflight_Priority_seat_comfort"
        ],
        "Travel Habits & Loyalty": [
            "Travel_Frequency_1 flight",
            "Travel_Frequency_2-3 flights",
            "Travel_Frequency_4-6 flights",
            "Travel_Frequency_7-10 flights",
            "Travel_Frequency_more than 10 flight",
            "Travel_Class_economy",
            "Travel_Class_first class",
            "Travel_Class_premium economy",
            "Price_Sensitivity_somewhat sensitive",
            "Price_Sensitivity_very sensitive",
            "Loyalty_Program_planning to join",
            "Loyalty_Program_yes"
        ]
    }

    default_groups = [
        "Travel Purpose",
        "Influencing Factors",
        "Reward Preferences",
        "Travel Habits & Loyalty"
    ]

    control_col1, control_col2, control_col3 = st.columns([1.3, 1, 1])

    with control_col1:
        selected_groups = st.multiselect(
            "Feature groups",
            options=list(feature_groups.keys()),
            default=default_groups
        )

    with control_col2:
        n_clusters = st.slider("Number of clusters", min_value=2, max_value=8, value=4)

    with control_col3:
        random_state = st.selectbox("Random seed", options=[42, 7, 21, 99], index=0)

    selected_features = [
        column
        for group in selected_groups
        for column in feature_groups[group]
        if column in df_seg.columns
    ]

    if len(selected_features) < 2:
        st.warning("Select at least one feature group with two or more columns to run clustering.")
    else:
        seg_input = df_seg[selected_features].copy()
        clusters, _, _ = run_kmeans(
            seg_input,
            n_clusters=n_clusters,
            random_state=random_state,
            n_init=10
        )

        seg_result = seg_input.copy()
        seg_result["Cluster"] = clusters
        seg_result["Customer_ID"] = np.arange(1, len(seg_result) + 1)

        components, explained_ratio = run_pca(seg_input, n_components=2)
        seg_result["PCA_1"] = components[:, 0]
        seg_result["PCA_2"] = components[:, 1]

        cluster_counts = (
            seg_result["Cluster"]
            .value_counts()
            .sort_index()
            .rename_axis("Cluster")
            .reset_index(name="Customers")
        )
        cluster_counts["Cluster Label"] = cluster_counts["Cluster"].apply(lambda x: f"Cluster {x}")
        cluster_counts["Share %"] = (cluster_counts["Customers"] / len(seg_result) * 100).round(1)

        profile_means = seg_result.groupby("Cluster")[selected_features].mean()
        overall_means = seg_input.mean()
        differentiators = profile_means.sub(overall_means, axis=1)

        top_traits = []
        for cluster_id in differentiators.index:
            strongest = differentiators.loc[cluster_id].abs().sort_values(ascending=False).head(3).index
            labels = [prettify_feature_name(feature) for feature in strongest]
            top_traits.append(", ".join(labels))

        profile_summary = cluster_counts.copy()
        profile_summary["Top differentiators"] = top_traits

        metric1, metric2, metric3 = st.columns(3)
        metric1.metric("Customers Segmented", len(seg_result))
        metric2.metric("Features Used", len(selected_features))
        metric3.metric("PCA Variance Explained", f"{(explained_ratio.sum() * 100):.1f}%")

        # Build a shared colour palette so both charts match
        _palette = px.colors.qualitative.Plotly
        unique_clusters = sorted(seg_result["Cluster"].unique())
        cluster_colors = {
            f"Cluster {c}": _palette[i % len(_palette)]
            for i, c in enumerate(unique_clusters)
        }

        seg_result["Cluster Label"] = seg_result["Cluster"].apply(lambda c: f"Cluster {c}")

        col1, col2 = st.columns([1, 1.4])

        with col1:
            fig_count = px.bar(
                cluster_counts,
                x="Cluster Label",
                y="Customers",
                color="Cluster Label",
                color_discrete_map=cluster_colors,
                text="Share %",
                title="Cluster Distribution"
            )
            fig_count.update_traces(texttemplate="%{text}%", textposition="outside")
            fig_count.update_layout(showlegend=False, yaxis_title="Customers", xaxis_title="")
            st.plotly_chart(fig_count, use_container_width=True)

        with col2:
            fig_pca = px.scatter(
                seg_result,
                x="PCA_1",
                y="PCA_2",
                color="Cluster Label",
                color_discrete_map=cluster_colors,
                hover_data={
                    "Customer_ID": True,
                    "Cluster": True,
                    "PCA_1": ":.2f",
                    "PCA_2": ":.2f"
                },
                title="Customer Segments in 2D PCA Space",
                labels={"Cluster Label": "Cluster"}
            )
            st.plotly_chart(fig_pca, use_container_width=True)

        st.subheader("Segment Profiles")
        for _, row in profile_summary.iterrows():
            st.markdown(
                f"**{row['Cluster Label']}** | Customers: {int(row['Customers'])} | "
                f"Share: {row['Share %']:.1f}% | Top differentiators: {row['Top differentiators']}"
            )

        focus_cluster = st.selectbox(
            "Inspect a cluster",
            options=sorted(seg_result["Cluster"].unique()),
            format_func=lambda x: f"Cluster {x}"
        )

        focus_profile = (
            profile_means.loc[focus_cluster]
            .sort_values(ascending=False)
            .head(10)
            .reset_index()
        )
        focus_profile.columns = ["Feature", "Average Score"]
        focus_profile["Feature"] = focus_profile["Feature"].apply(prettify_feature_name)

        fig_profile = px.bar(
            focus_profile.sort_values("Average Score"),
            x="Average Score",
            y="Feature",
            orientation="h",
            color="Average Score",
            title=f"Top Features for Cluster {focus_cluster}"
        )
        st.plotly_chart(fig_profile, use_container_width=True)

        heatmap_data = profile_means.copy()
        heatmap_data.index = [f"Cluster {idx}" for idx in heatmap_data.index]
        top_heatmap_features = differentiators.abs().mean().sort_values(ascending=False).head(10).index.tolist()

        fig_heatmap = px.imshow(
            heatmap_data[top_heatmap_features],
            aspect="auto",
            color_continuous_scale="Blues",
            labels={"x": "Features", "y": "Cluster", "color": "Avg Score"},
            title="Cluster Comparison Heatmap"
        )
        fig_heatmap.update_xaxes(tickangle=-35)
        st.plotly_chart(fig_heatmap, use_container_width=True)

        st.info("**Interactive Insight:**\n\nModify the feature groups or cluster count to see how customer segments shift. This helps compare whether airline strategy should focus more on travel purpose, service quality, or loyalty behavior.")



# =================================================
# 7. AIRLINE-SPECIFIC DRIVERS
# =================================================
elif page == "Airline-Specific Drivers":

    st.subheader("Interactive Airline-Specific Drivers")

    st.markdown("""
    Explore the strongest **airline-specific customer drivers** from `airline_model_ready.csv`.
    This compares each selected airline against all others and highlights the features that stand out most.
    """)

    df_model = load_model_ready_data()

    airline_mapping = {
        0: "Air India",
        1: "Akasa",
        2: "Indigo",
        3: "SpiceJet",
        4: "Vistara"
    }

    feature_groups = {
        "Passenger Profile": [
            "Gender", "Age", "Occupation", "Travel_Frequency", "Travel_Class",
            "Flight_Preference", "Booking_Mode", "Price_Sensitivity",
            "Loyalty_Program", "Schedule_Preference"
        ],
        "Travel Purpose": [
            "business", "leisure", "family_visit", "education", "medical"
        ],
        "Influencing Factors": [
            "Influencing_Factors_brand_reputation",
            "Influencing_Factors_free_food",
            "Influencing_Factors_in_flight_service_quality",
            "Influencing_Factors_loyalty_programs",
            "Influencing_Factors_punctuality",
            "Influencing_Factors_safety_rating",
            "Influencing_Factors_seat_comfort",
            "Influencing_Factors_ticket_price"
        ],
        "Reward Preferences": [
            "Reward_Preference_cashback_or_discount",
            "Reward_Preference_extra_baggage",
            "Reward_Preference_free_flights",
            "Reward_Preference_free_lounge_access",
            "Reward_Preference_free_seat_upgrades",
            "Reward_Preference_priority_check-in_boarding"
        ],
        "In-Flight Priorities": [
            "Inflight_Priority_amenitiese.g_charging_ports_eye-mask_blankets",
            "Inflight_Priority_cabin_crew_behavior",
            "Inflight_Priority_extra_baggage_facility",
            "Inflight_Priority_extra_legroom",
            "Inflight_Priority_in-flight_entertainment",
            "Inflight_Priority_seat_comfort"
        ]
    }

    control_col1, control_col2, control_col3 = st.columns([1.4, 1.2, 1])

    with control_col1:
        selected_airlines = st.multiselect(
            "Airlines to compare",
            options=list(airline_mapping.keys()),
            default=[2, 3, 4],
            format_func=lambda x: airline_mapping[x]
        )

    with control_col2:
        selected_driver_groups = st.multiselect(
            "Feature groups",
            options=list(feature_groups.keys()),
            default=["Passenger Profile", "Travel Purpose", "Influencing Factors", "Reward Preferences"]
        )

    with control_col3:
        top_n = st.slider("Top drivers", min_value=5, max_value=12, value=8)

    selected_features = [
        column
        for group in selected_driver_groups
        for column in feature_groups[group]
        if column in df_model.columns
    ]

    if not selected_airlines:
        st.warning("Select at least one airline to view driver analysis.")
    elif not selected_features:
        st.warning("Select at least one feature group to continue.")
    else:
        airline_counts = (
            df_model["Airline_Selected"]
            .value_counts()
            .sort_index()
            .rename_axis("Airline_ID")
            .reset_index(name="Passengers")
        )
        airline_counts["Airline"] = airline_counts["Airline_ID"].map(airline_mapping)

        metric1, metric2, metric3 = st.columns(3)
        metric1.metric("Airlines Selected", len(selected_airlines))
        metric2.metric("Features Compared", len(selected_features))
        metric3.metric("Rows Used", len(df_model))

        fig_distribution = px.pie(
            airline_counts,
            names="Airline",
            values="Passengers",
            hole=0.45,
            title="Airline Sample Distribution"
        )
        st.plotly_chart(fig_distribution, use_container_width=True)

        summary_rows = []
        heatmap_rows = []

        for airline_id in selected_airlines:
            airline_name = airline_mapping.get(airline_id, f"Airline {airline_id}")
            scores = compute_one_vs_rest_driver_scores(
                df_model,
                target_col="Airline_Selected",
                airline_id=airline_id,
                selected_features=selected_features
            ).head(top_n)

            if scores.empty:
                continue

            scores_display = scores.copy()
            scores_display["Feature"] = scores_display["Feature"].apply(prettify_feature_name)
            scores_display["Driver Score"] = scores_display["Driver Score"].round(3)
            scores_display["Airline Avg"] = scores_display["Airline Avg"].round(3)
            scores_display["Others Avg"] = scores_display["Others Avg"].round(3)

            summary_rows.append({
                "Airline": airline_name,
                "Strongest Driver": scores_display.iloc[0]["Feature"],
                "Driver Score": scores_display.iloc[0]["Driver Score"]
            })

            airline_heatmap = scores_display[["Feature", "Driver Score"]].copy()
            airline_heatmap["Airline"] = airline_name
            heatmap_rows.append(airline_heatmap)

            st.subheader(f"Top Drivers for {airline_name}")

            fig_driver = px.bar(
                scores_display.sort_values("Driver Score"),
                x="Driver Score",
                y="Feature",
                orientation="h",
                color="Direction",
                title=f"{airline_name}: Top {top_n} Drivers"
            )
            st.plotly_chart(fig_driver, use_container_width=True)

            for _, row in scores_display.iterrows():
                st.markdown(
                    f"- **{row['Feature']}** | Driver Score: `{row['Driver Score']}` | "
                    f"{row['Direction']} | Airline Avg: `{row['Airline Avg']}` | Others Avg: `{row['Others Avg']}`"
                )

        if summary_rows:
            st.subheader("Airline Driver Summary")
            for row in summary_rows:
                st.markdown(
                    f"**{row['Airline']}** | Strongest Driver: {row['Strongest Driver']} | "
                    f"Driver Score: {row['Driver Score']}"
                )

            summary_df = pd.DataFrame(summary_rows)
            fig_summary = px.bar(
                summary_df,
                x="Airline",
                y="Driver Score",
                color="Strongest Driver",
                title="Strongest Driver by Airline"
            )
            st.plotly_chart(fig_summary, use_container_width=True)

            heatmap_df = pd.concat(heatmap_rows, ignore_index=True)
            heatmap_pivot = heatmap_df.pivot(index="Airline", columns="Feature", values="Driver Score").fillna(0)

            st.subheader("Cross-Airline Driver Comparison")
            fig_heatmap = px.imshow(
                heatmap_pivot,
                aspect="auto",
                color_continuous_scale="Sunset",
                labels={"x": "Feature", "y": "Airline", "color": "Driver Score"},
                title="Driver Strength Heatmap"
            )
            fig_heatmap.update_xaxes(tickangle=-35)
            st.plotly_chart(fig_heatmap, use_container_width=True)
