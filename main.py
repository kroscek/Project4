import os

os.chdir('/home/lemma/MNIST-GUI')
from tkinter import *
import pyscreenshot as ImageGrab
import cnn
import time


class MNIST_GUI:
    def __init__(self, root):
        self.root = root
        self.res = ""
        self.pre = [None, None]
        self.model = cnn.Model()
        self.bs = 8.5
        self.c = Canvas(self.root, bd=3, relief="ridge", width=300, height=282, bg='white')
        self.c.grid(row=0, columnspan=5)
        self.c.pack(side=LEFT)
        f1 = Frame(self.root, padx=5, pady=5)
        Label(f1, text="Real-Time Hand-Written Digits Recognition", fg="green", font=("", 15, "bold")).pack(pady=10)
        Label(f1, text="<<--Draw Your Digit on Canvas", fg="green", font=("", 15)).pack()
        self.pr = Label(f1, text="Prediction: None", fg="blue", font=("", 20, "bold"))
        self.pr.pack(pady=20)
        self.predictionScores = Text(f1, height=10, width=25, padx=5, bg='white',
                                     borderwidth=4, highlightthickness=0,
                                     relief='ridge')
        self.predictionScores.pack(side=RIGHT)
        ll = Label(f1, text="Scores:", fg="black", font=("", 15, "bold"))
        ll.place(relx=0.7, rely=0.45)
        Button(f1, font=("", 15), fg="white", bg="red", text="Clear", command=self.clear).pack(side=BOTTOM)

        f1.pack(side=RIGHT, fill=Y)
        self.c.bind("<Button-1>", self.putPoint)
        self.c.bind("<ButtonRelease-1>", self.result)
        self.c.bind("<B1-Motion>", self.paint)

    def result(self, e):
        x = self.root.winfo_rootx() + self.c.winfo_x()
        y = self.root.winfo_rooty() + self.c.winfo_y()
        x1 = x + self.c.winfo_width()
        y1 = y + self.c.winfo_height()
        img = ImageGrab.grab()
        img = img.crop((x, y, x1, y1))
        img.save("dist.png")
        prediction, scores = self.model.Predict("dist.png")
        self.pr['text'] = "Prediction: " + str(prediction)
        n = 0
        for score in scores:
            self.predictionScores.insert(END, "{}: {}\n".format(n, score))
            n += 1

    def clear(self):
        self.c.delete('all')
        self.predictionScores.delete(1.0, END)

    def putPoint(self, e):
        self.c.create_oval(e.x - self.bs, e.y - self.bs, e.x + self.bs, e.y + self.bs, outline='black', fill='black')
        self.pre = [e.x, e.y]

    def paint(self, e):
        self.c.create_line(self.pre[0], self.pre[1], e.x, e.y, width=self.bs * 2, fill='black', capstyle=ROUND,
                           smooth=TRUE)

        self.pre = [e.x, e.y]


if __name__ == "__main__":
    root = Tk()
    MNIST_GUI(root)
    root.title('Real-Time Hand-Written Digit Recognition')
    root.resizable(0, 0)
    root.mainloop()
