import streamlit as st
import pandas as pd
import hmac
import pickle

st.set_page_config(page_title="Marketing Mix Optimizer", layout="wide")

def check_password():
    """Returns `True` if the user has entered the correct password."""
    def password_entered():
        """Checks whether the password entered by the user is correct."""
        if st.session_state["username"] in st.secrets["passwords"] and hmac.compare_digest(
            st.session_state["password"], st.secrets["passwords"][st.session_state["username"]]
        ):
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # remove password from session state for security
            del st.session_state["username"]  # remove username from session state for security
        else:
            st.session_state["password_correct"] = False

    # First time running, show the login form
    if "password_correct" not in st.session_state:
        with st.form("login_form"):
            st.text_input("Username", key="username")
            st.text_input("Password", type="password", key="password")
            submitted = st.form_submit_button("Log in")
            if submitted:
                password_entered()

    # If the password is not correct, show an error message
    if not st.session_state.get("password_correct"):
        st.error("User not known or password incorrect⚠️")
        return False

    return True

# Ensure the password check only happens once and stops execution if not logged in
if not check_password():
    st.stop()

def main():
    st.title("Marketing Mix Optimizer")

    # Sidebar for navigation
    st.sidebar.title("Please select a page..")
    page = st.sidebar.radio("Choose", ["Model Discovery", "Explainability", "Recommendation"])

    if page == "Model Discovery":
        # File uploader
        uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.session_state.df = df

                st.write("Dataset:")
                st.write(df.head())
                st.header("Model Selection:")
                st.image("Model_comparison.jpeg")
            except Exception as e:
                st.error(f"Error reading the file: {e}")

    elif page == "Explainability":
        st.header("Model Visualization")

        # Display Feature Importance at the top
        st.image("Feature_imp.jpeg", caption="Feature Importance", use_column_width=True)

        # Create two columns for VOC and Error Analysis images
        col1, col2 = st.columns([1, 1])

        with col1:
            st.image("VOC.jpeg", caption="VOC", use_column_width=True)

        with col2:
            st.image("error.jpeg", caption="Error Analysis", use_column_width=True)

    elif page == "Recommendation":
        st.header("Recommendation System")

        # File uploader for new data
        file_path = st.file_uploader("Choose a CSV file", type=["csv"])

        if file_path is not None:
            try:
                X_data1 = pd.read_csv(file_path)

                # One-hot encoding for categorical variables
                df_ohe = X_data1[['Marketplace', 'Ad_group']]
                data_encoded = pd.get_dummies(df_ohe, columns=['Marketplace', 'Ad_group'])
                data_encoded = data_encoded.applymap(lambda x: 1 if x else 0)

                # Combine the encoded data with the original data
                final_data = pd.concat([data_encoded, X_data1], axis=1)

                # Prepare test data
                X_test1 = final_data.drop(['Marketplace', 'Ad_group'], axis=1)

                # Load the model
                loaded_model = pickle.load(open('model.pkl', 'rb'))

                # Make predictions
                y_pred1 = loaded_model.predict(X_test1)
                new_df = pd.DataFrame(y_pred1, columns=['Sales'])

                # Combine predictions with the original data
                X_new = pd.concat([final_data, new_df], axis=1, join='inner')

                # Pivot table for analysis
                pivot_table = pd.pivot_table(X_new, values=['Sales', 'Spend'], index=['Ad_group', 'Marketplace'], aggfunc='sum')
                pivot_table.reset_index(inplace=True)
                pivot_table['ROI'] = (pivot_table['Sales'] - pivot_table['Spend']) / pivot_table['Spend']
                pivot_table['rank'] = pivot_table.groupby('Marketplace')['ROI'].rank(ascending=False)

                # Display the pivot table and the best performing rows
                st.write(pivot_table)
                st.header("Final Recommendation for optimum budget mobilisation for underlying MarketPlace:")
                st.write(pivot_table[pivot_table["rank"] == 1.0])
                

            except Exception as e:
                st.error(f"Error processing the file: {e}")

if __name__ == "__main__":
    main()
