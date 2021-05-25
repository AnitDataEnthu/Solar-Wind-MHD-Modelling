import smtplib

# list of email_id to send the mail
li=["anitg.4@gmail.com","sonamdawani@gmail.com"]
model_number=1
for dest in li:
    s=smtplib.SMTP('smtp.gmail.com',587)
    s.starttls()
    s.login("anitg.4@gmail.com","9ijn(IJN")
    message="Updating you with the model training progress:"+str(model_number)+"completed"
    s.sendmail("sender_email_id",dest,message)
    s.quit()