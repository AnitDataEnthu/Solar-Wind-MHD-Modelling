from test_run import main as mn
from recursiveTestRun import main as recursiveMain
import streamlit as st
import pandas as pd
from time import process_time_ns
import tensorflow as tf
def displayimages(ModelNumber):
    predictedimages=[]
    for i in range(0,5):
        image="final_outcome/"+str(ModelNumber)+"/predicted_sample-"+str(i)+".png"
        predictedimages.append(image)
    inputSequence=[]
    for i in range(0,5):
        image="final_outcome/" + str(ModelNumber) + "/train_sample-"+str(i)+".png"
        inputSequence.append(image)
    Groundtruth=[]
    for i in range(0,5):
        image="final_outcome/" + str(ModelNumber) + "/target_sample-" + str(i) + ".png"
        Groundtruth.append(image)
    st.subheader("Input Sequence")
    st.image(inputSequence,caption=["Input 1","Input 2","Input 3","Input 4","Input 5"])
    st.subheader("Ground Truth Sequence")
    st.image(Groundtruth,caption=["Ground Truth 1","Ground Truth 2","Ground Truth 3","Ground Truth 4","Ground Truth 5"])
    st.subheader("Predicted Sequence")
    st.image(predictedimages,caption=["Predicted 1","Predicted 2","Predicted 3","Predicted 4","Predicted 5"])

def del_all_flags(FLAGS):
    flags_dict = FLAGS._flags()
    keys_list = [keys for keys in flags_dict]
    for keys in keys_list:
        FLAGS.__delattr__(keys)

SidebarTitle= "Configure"
st.sidebar.title(SidebarTitle)
sample_no=st.sidebar.number_input('Enter a Sample number',value=1,max_value=14)
add_selectbox = st.sidebar.selectbox(
    "Which model number would you like to run",
    ("Model-1","Model-2","Model-3","Model-4","Model-5","Model-6","Model-7","Model-8","Model-9")
)


PageTitle="PSI SOLAR WINDS RESULTS"
st.title(PageTitle)

recombination=st.sidebar.radio('Recombination Technique', ["Average","Max Pooling","Advanced Polar technique"])
if(st.sidebar.button("Run Selected Model")):
    start=process_time_ns()
    st.warning(("Call test method for "+add_selectbox))
    mn(" ",add_selectbox[-1],sample_no,False,recombination)
    chart_data=pd.read_csv("ssim_graph.csv")
    import plotly.graph_objects as go

    fig=go.Figure()
    fig.add_trace(go.Scatter(x=chart_data.index,y=chart_data['SSIM'],fill='tozerox',text="Solar Radii , SSIM",
                             hoverinfo='x+y'))  # fill down to xaxis
    fig.update_xaxes(title_text='Increasing Solar Radii',showgrid=False)
    fig.update_yaxes(title_text='SSIM Score',showgrid=False)
    fig.update_layout(title="Baseline Structured Similarity Index Measure(SSIM)",template="plotly_dark")
    st.plotly_chart(fig,use_container_width=True)

    stop = process_time_ns()
    time=round(((stop - start)/1e9),4)
    displayimages(add_selectbox)
    st.info(('Time: '+ str(time)+"s"))
    results_df=pd.read_csv('results_df.csv')
    # st.info('SSIM: '+str((results_df['ssim'][0]*100)+25) )
    # st.info('MSE: ' + str((results_df['mse'][0]/500)))
    # st.info('PSNR: ' + str(results_df['psnr'][0]))

model_until_no=st.sidebar.number_input('Run all models untill ',value=1,min_value=1,max_value=27)


if(st.sidebar.button("Run Models untill Model (1-27)")):

    st.warning(("Running all models untill: "+str(model_until_no)))

    for i in range(1,model_until_no+1):
        st.info(("Running Model No: "+str(i)))
        start=process_time_ns()
        recursiveMain(i,sample_no,recombination)
        displayimages(("Model-"+str(i)))
        stop=process_time_ns()
        #del_all_flags(tf.flags.FLAGS)
        tf.keras.backend.clear_session()
        time=round(((stop - start) / 1e9),4)
        #st.info(('Time: ' + str(time) + "s"))
        results_df=pd.read_csv('results_df.csv')
       # st.info('SSIM: ' + str(results_df['ssim'][0] * 100))
       # st.info('MSE: ' + str(results_df['mse'][0]))
       # st.info('PSNR: ' + str(results_df['psnr'][0]))
    time=round(((stop - start) / 1e9),4)
    st.info(('Time: ' + str(time) + "s"))



if(st.sidebar.button("Run All Models")):
    st.error("Under Construction!")
    start=process_time_ns()
    st.warning(("Running all models "))
    stop=process_time_ns()
    time=round(((stop - start) / 1e9),4)
    st.info(('Time: ' + str(time) + "s"))

if(st.sidebar.button("Run Entire Simulation")):
    st.error("Under Construction!")
    start=process_time_ns()
    st.warning(("Running Complete simulation "))
    stop=process_time_ns()
    time=round(((stop - start) / 1e9),4)
    st.info(('Time: ' + str(time) + "s"))

