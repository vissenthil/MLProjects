from tkinter import *


def Show_entry():
    p1 = float(e1.get())
    p2 = float(e2.get())
    p3 = float(e3.get())
    p4 = float(e4.get())
    p5 = float(e5.get())
    p6 = float(e6.get())
    print(p1)
def app_GUI():
    global e1
    global e2
    global e3
    global e4
    global e5
    global e6

    master = Tk()
    master.title('Insurance Cost Prediction')

    label = Label(master, text='Insurance Cost Prediction', bg='black', fg='white').grid(row=0, columnspan=2)
    label = Label(master, text="Enter your age").grid(row=1)
    label = Label(master, text='Male or female [1/0]').grid(row=2)
    label = Label(master, text='Enter your BMI Value').grid(row=3)
    label = Label(master, text='Enter No of children').grid(row=4)
    label = Label(master, text='Smoker Yes or No [1/0]').grid(row=5)
    label = Label(master, text='Region [1-4]').grid(row=6)
    e1 = Entry(master)
    e2 = Entry(master)  # , textvariable=first_2).grid(row=2, column=1)
    e3 = Entry(master)  # , textvariable=first_3).grid(row=3, column=1)
    e4 = Entry(master)
    e5 = Entry(master)
    e6 = Entry(master)
    e1.grid(row=1,column=1)
    e2.grid(row=2, column=1)
    e3.grid(row=3, column=1)
    e4.grid(row=4, column=1)
    e5.grid(row=5, column=1)
    e6.grid(row=6, column=1)



    Button(master, text='Predict', command=Show_entry).grid()
    master.mainloop()
        #print(f'IN1 = {In1}, In2 = {In2}, In3 = {In3},In4 = {In4},In5 = {In5},In6 = {In6}')

if __name__ == '__main__':
    app_GUI()
