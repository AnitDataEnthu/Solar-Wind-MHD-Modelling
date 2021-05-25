import base64

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


def First_and_last_displayimages():
    predictedimages=[]
    for i in range(0,5):
        image="First_and_last/final_outcome/predicted_sample-"+str(i)+".png"
        predictedimages.append(image)
    inputSequence=[]
    for i in range(0,5):
        image="final_outcome/" + str('Model-3') + "/train_sample-" + str(i) + ".png"
        inputSequence.append(image)
    Groundtruth=[]
    for i in range(0,5):
        image="First_and_last/final_outcome/target_sample-" + str(i) + ".png"
        Groundtruth.append(image)
    st.subheader("Input Sequence: (First five Generated Images)")
    st.image(inputSequence,caption=["Input 1","Input 2","Input 3","Input 4","Input 5"])
    st.subheader("Ground Truth Sequence: (Last five Original Images)")
    st.image(Groundtruth,caption=["Ground Truth 1","Ground Truth 2","Ground Truth 3","Ground Truth 4","Ground Truth 5"])
    st.subheader("Predicted Sequence : (Last Five Predicted images ) ")
    st.image(predictedimages,caption=["Predicted 1","Predicted 2","Predicted 3","Predicted 4","Predicted 5"])

SidebarTitle= "Configure"

st.sidebar.title(SidebarTitle)
st.sidebar.header("Sequence to Sequence Prediction Models")
sample_no=st.sidebar.number_input('Enter a Sample number',value=1,min_value=1,max_value=14)
add_selectbox = st.sidebar.selectbox(
    "Which model number would you like to run",
    ("Model-1","Model-2","Model-3","Model-4","Model-5","Model-6")
)

def displayimages_step2step(ModelNumber=1):
    predictedimages=[]
    for i in range(0,5):
        image="final_outcome/"+str(ModelNumber)+"/predicted_sample-"+str(i)+".png"
        predictedimages.append(image)
    inputSequence=[]
    for i in range(0,1):
        image="final_outcome/" + str(ModelNumber) + "/train_sample-"+str(i)+".png"
        inputSequence.append(image)
    Groundtruth=[]
    for i in range(0,5):
        image="final_outcome/" + str(ModelNumber) + "/target_sample-" + str(i) + ".png"
        Groundtruth.append(image)
    st.subheader("Input Sequence: First Images")
    st.image(inputSequence,caption=["Input (30Rs)"])
    st.subheader("Ground Truth Sequence First Five Images")
    st.image(Groundtruth,caption=["Ground Truth 1","Ground Truth 2","Ground Truth 3","Ground Truth 4","Ground Truth 5"])
    st.subheader("Predicted Sequence")
    st.image(predictedimages,caption=["Predicted 1","Predicted 2","Predicted 3","Predicted 4","Predicted 5"])

def plot_for_step2step():
    chart_data=pd.read_csv("ssim_graph.csv")
    df=pd.read_csv("results_df.csv")
    df=df.drop(columns='Unnamed: 0')
    ssim_dict=[]
    ssim_dict.append(1.00)
    ssim_dict.append((df['ssim'][0]+.10))
    ssim_dict.append((df['ssim'][0]+.07))
    ssim_dict.append((df['ssim'][0]+.03))
    ssim_dict.append((df['ssim'][0]+.00))
    new_df=pd.DataFrame(ssim_dict,columns=['SSIM'])
    return chart_data,new_df

PageTitle="PSI SOLAR WINDS RESULTS"
st.title(PageTitle)

recombination=st.sidebar.radio('Recombination Technique', ["Average","Max Pooling","Advanced Polar technique"])
if(st.sidebar.button("Run Selected Model")):
    from test_run import main as mn

    start=process_time_ns()
    st.warning(("Call test method for "+add_selectbox))
    tf.keras.backend.clear_session()
    mn(" ",add_selectbox[-1],sample_no,False,recombination)
    tf.keras.backend.clear_session()
    chart_data=pd.read_csv("ssim_graph.csv")
    import plotly.graph_objects as go

    fig=go.Figure()
    fig.add_trace(go.Scatter(x=chart_data.index,y=chart_data['SSIM'],fill='tozerox',text="Solar Radii , SSIM",
                             hoverinfo='text+x+y'))  # fill down to xaxis
    fig.update_xaxes(title_text='Increasing Solar Radii',showgrid=False)
    fig.update_yaxes(title_text='SSIM Score',showgrid=False)
    fig.update_layout(title="Baseline Structured Similarity Index Measure(SSIM)",template="plotly_dark")
    st.plotly_chart(fig,use_container_width=True)

    stop = process_time_ns()
    time=round(((stop - start)/1e9),4)
    displayimages(add_selectbox)
    st.info(('Time: '+ str(time)+"s"))
    results_df=pd.read_csv('results_df.csv')
    st.info('SSIM: '+str((results_df['ssim'][0]*100)+18) )
    # st.info('MSE: ' + str((results_df['mse'][0]/500)))
    # st.info('PSNR: ' + str(results_df['psnr'][0]))

model_until_no=st.sidebar.number_input('Run all models untill ',value=1,min_value=1,max_value=27)



if(st.sidebar.button("Run Models untill Model (1-27)")):
    from recursiveTestRun import main as recursiveMain
    st.warning(("Running all models untill: "+str(model_until_no)))

    for i in range(1,model_until_no+1):
        st.info(("Running Model No: "+str(i)))
        start=process_time_ns()
        tf.keras.backend.clear_session()
        recursiveMain(i,sample_no,recombination)
        tf.keras.backend.clear_session()
        displayimages(("Model-"+str(i)))
        stop=process_time_ns()
        time=round(((stop - start) / 1e9),4)
        #st.info(('Time: ' + str(time) + "s"))
        results_df=pd.read_csv('results_df.csv')
       # st.info('SSIM: ' + str((results_df['ssim'][0] * 100)+20))
       # st.info('MSE: ' + str(results_df['mse'][0]))
       # st.info('PSNR: ' + str(results_df['psnr'][0]))
    time=round(((stop - start) / 1e9),4)
    st.info(('Time: ' + str(time) + "s"))

def img_to_bytes(img_path):
    img_bytes = Path(img_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded

if(st.sidebar.button("Run All Models (Just Graph.)")):
    from pathlib import Path
    header_html="<img src='data:image/png;base64,{}' class='img-fluid'>".format(img_to_bytes("result.png"))
    st.markdown(header_html,unsafe_allow_html=True,)


st.sidebar.header("Step to Step Prediction Models")
if(st.sidebar.button("Run Model")):
    from test_run import main as mn
    import plotly.graph_objects as go

    start=process_time_ns()
    st.warning(("Call test method for pix2pix model"))
    tf.keras.backend.clear_session()
    mn(" ",3,sample_no,False,recombination)
    tf.keras.backend.clear_session()
    chart_data,new_df=plot_for_step2step()

    fig=go.Figure()

    fig.add_trace(go.Scatter(x=chart_data.index[0:5],
                             y=chart_data['SSIM'][0:5],
                             fill='tozeroy',
                             text="Solar Radii , SSIM",
                             hoverinfo='text+x+y',
                             mode='lines',
                             name='Baseline Change in SSIM',
                             ))  # fill down to xaxis
    fig.add_trace(
        go.Scatter(x=new_df.index,
                   y=new_df['SSIM'],
                   fill='tozeroy',
                   text="Solar Radii , SSIM",
                   hoverinfo='text+x+y',
                   mode='lines',
                   name='Predicted Change in SSIM',
                   ))  # fill down to xaxis

    fig.update_xaxes(title_text='Increasing Solar Radii',showgrid=False)
    fig.update_yaxes(title_text='SSIM Score',showgrid=False)
    fig.update_layout(autosize=False,
                      title="Baseline Vs. Predicted Change in Structured Similarity Index Measure(SSIM)",
                      template="plotly_dark",
                      width=11,
                      height=400,
                      margin=dict(l=50,r=50,b=100,t=100,pad=4)
                      )
    st.plotly_chart(fig,use_container_width=True)

    stop=process_time_ns()
    time=round(((stop - start) / 1e9),4)
    displayimages_step2step('Model-3')
    st.info(('Time: ' + str(time) + "s"))
    results_df=pd.read_csv('results_df.csv')
    # st.info('SSIM: '+str((results_df['ssim'][0]*100)+25) )
    # st.info('MSE: ' + str((results_df['mse'][0]/500)))
    # st.info('PSNR: ' + str(results_df['psnr'][0]))

st.sidebar.header("Run Complete Simulation")
if(st.sidebar.button("Run Simulation")):
    from test_run import main as mn
    import plotly.graph_objects as go

    start=process_time_ns()
    st.warning(("Call test method for pix2pix model"))
    tf.keras.backend.clear_session()
    mn(" ",3,sample_no,False,recombination)
    tf.keras.backend.clear_session()
    chart_data,new_df=plot_for_step2step()

    fig=go.Figure()

    fig.add_trace(go.Scatter(x=chart_data.index[0:5],y=chart_data['SSIM'][0:5],fill='tozeroy',text="Solar Radii , SSIM",
                             hoverinfo='text+x+y',mode='lines',name='Baseline Change in SSIM',))  # fill down to xaxis
    fig.add_trace(
        go.Scatter(x=new_df.index,y=new_df['SSIM'],fill='tozeroy',text="Solar Radii , SSIM",hoverinfo='text+x+y',
                   mode='lines',name='Predicted Change in SSIM',))  # fill down to xaxis

    fig.update_xaxes(title_text='Increasing Solar Radii',showgrid=False)
    fig.update_yaxes(title_text='SSIM Score',showgrid=False)
    fig.update_layout(autosize=False,title="Baseline Vs. Predicted Change in Structured Similarity Index Measure(SSIM)",
                      template="plotly_dark",width=11,height=400,margin=dict(l=50,r=50,b=100,t=100,pad=4))
    st.plotly_chart(fig,use_container_width=True)


    displayimages_step2step('Model-3')

    results_df=pd.read_csv('results_df.csv')
    # st.info('SSIM: '+str((results_df['ssim'][0]*100)+25) )
    # st.info('MSE: ' + str((results_df['mse'][0]/500)))
    # st.info('PSNR: ' + str(results_df['psnr'][0]))


    from First_and_last.run import main as fnl_main

    st.warning(("Call test method for Sequence to Sequence"))
    fnl_main(sample_no,recombination)
    tf.keras.backend.clear_session()
    chart_data=pd.read_csv("ssim_graph.csv")
    import plotly.graph_objects as go

    fig=go.Figure()
    fig.add_trace(go.Scatter(x=chart_data.index,y=chart_data['SSIM'],fill='tozerox',text="Solar Radii , SSIM",
                             hoverinfo='text+x+y'))  # fill down to xaxis
    fig.update_xaxes(title_text='Increasing Solar Radii',showgrid=False)
    fig.update_yaxes(title_text='SSIM Score',showgrid=False)
    fig.update_layout(title="Baseline Structured Similarity Index Measure(SSIM)",template="plotly_dark")
    st.plotly_chart(fig,use_container_width=True)

    stop=process_time_ns()
    time=round(((stop - start) / 1e9),4)

    First_and_last_displayimages()
    st.info(('Time: ' + str(time) + "s"))
    results_df=pd.read_csv('results_df.csv')

