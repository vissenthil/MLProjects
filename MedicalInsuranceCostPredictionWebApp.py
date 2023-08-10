import streamlit as st
import joblib


def main():
    Html_temp = '''
    <div style ="background-color:lightblue;padding:16px">
    <h2 stype= "color:black";text align:center > Health Insurance Cost Prediction using ML </h2>
    </div> 
    '''
    st.markdown(Html_temp,unsafe_allow_html=True)
    model = joblib.load('D:\\python\\MLProjects\\Medical_cost_prediction\\MedicalInsuranceCostPrediction\\LR_Ins_Cost_prediction')
    p1 = st.slider('Enter your age',18,100)
    s1 = st.selectbox('sex',('Male','Female'))
    if s1 == 'Male':
        p2 = 1
    else:
        p2 = 0

    p3 = st.number_input('Enter BMI Value')
    p4 = st.slider('Plese enter number of children',0,5)
    s2 = st.selectbox('Smoker',('Yes','No'))

    if s2 == 'Yes':
        p5 = 1
    else:
        p5 = 0
    p6 = st.slider('Enter the Region',0,4)

     # LR_Ins_Cost_prediction Model name
    if st.button('Predict'):
       pred = model.predict([[p1,p2,p3,p4,p5,p6]])
       st.success('Your insurance cost is {}'.format(round(pred[0],3)))

#python -m streamlit <command> use this for running streamlit in pycharm
#How to run from the command prompt --Need to run from the command prompt like this
#C:\Users\viss>python -m streamlit run


if __name__ == "__main__":
    main()
