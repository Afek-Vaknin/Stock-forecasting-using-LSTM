import socket
from tkinter import *
from tkinter import messagebox
import string

HOST = "127.0.0.1"
PORT = 52744
global client


class Window:
    def __init__(self, size):
        self._root = Tk()
        self._root.title("Stocker")
        self._root.geometry(size)
        self._root.configure(bg="white")
        self._root.resizable(False, False)
        self.center(self._root)

    def center(self, win):
        """
        centers a tkinter window
        :param win: the main window or Toplevel window to center
        """
        win.update_idletasks()
        width = win.winfo_width()
        frm_width = win.winfo_rootx() - win.winfo_x()
        win_width = width + 2 * frm_width
        height = win.winfo_height()
        titlebar_height = win.winfo_rooty() - win.winfo_y()
        win_height = height + titlebar_height + frm_width
        x = win.winfo_screenwidth() // 2 - win_width // 2
        y = win.winfo_screenheight() // 2 - win_height // 2
        win.geometry('{}x{}+{}+{}'.format(width, height, x, y))

    def move_to_login(self,):
        self._root.destroy()
        LoginPage()

    def move_to_signup(self,):
        self._root.destroy()
        SignUpPage()

    def move_to_home(self,):
        self._root.destroy()
        HomePage()

    def move_to_main_page(self):
        self._root.destroy()
        MainPage()


class HomePage(Window):
    def __init__(self):
        # define the main frame
        Window.__init__(self, "400x500")
        # define the secondary frame
        self._frame = Frame(self._root, width=350, height=500, bg="white")
        self._frame.place(x=25, y=50)

        #
        self._heading = Label(self._frame, text="Home Page", fg="#57a1f8",
                              bg="white", font=("Microsoft YaHei UI Light", 23, "bold"))
        self._heading.place(x=86, y=5)

        # move to log in
        self._log_in = Button(self._frame, width=10, text="Log in", border=0, bg="#57a1f8",
                              cursor="hand2", fg="white", font=("Microsoft YaHei UI Light", 23, "bold"),
                              command=self.move_to_login)
        self._log_in.place(x=77, y=95)

        # move to sign up
        self._sign_up = Button(self._frame, width=10, text="Sign up", border=0, bg="#57a1f8",
                               cursor="hand2", fg="white", font=("Microsoft YaHei UI Light", 23, "bold"),
                               command=self.move_to_signup)
        self._sign_up.place(x=77, y=210)

        self._root.mainloop()


class LoginPage(Window):
    def __init__(self):
        # define the main frame
        Window.__init__(self, "400x500")

        # define the secondary frame
        self._frame = Frame(self._root, width=350, height=500, bg="white")
        self._frame.place(x=25, y=50)

        self._heading = Label(self._frame, text="Login Page", fg="#57a1f8",
                              bg="white", font=("Microsoft YaHei UI Light", 23, "bold"))
        self._heading.place(x=80, y=5)

        # receiving username block
        self._label = Label(self._frame, text="Username:", fg="black",
                            bg="white", font=("Microsoft YaHei UI Light", 11))
        self._label.place(x=25, y=85)
        self._user = Entry(self._frame, width=25, fg="black", border=0,
                           bg="white", font=("Microsoft YaHei UI Light", 11))
        self._user.place(x=27, y=115)
        Frame(self._frame, width=295, height=2, bg="black").place(x=25, y=137)

        # receiving password block
        self._label = Label(self._frame, text="Password:", fg="black",
                            bg="white", font=("Microsoft YaHei UI Light", 11))
        self._label.place(x=25, y=170)
        self._code = Entry(self._frame, width=25, fg="black", border=0,
                           bg="white", font=("Microsoft YaHei UI Light", 11), show='*')
        self._code.place(x=27, y=195)
        Frame(self._frame, width=295, height=2, bg="black").place(x=25, y=217)

        # password hiding
        self._eye_picture = PhotoImage(file="close_eye.png")
        self._eye_button = Button(self._frame, image=self._eye_picture, bd=0, bg="white",
                                  activebackground="white", cursor="hand2", command=self.show_password)
        self._eye_button.place(x=295, y=191)

        # Move to site page button
        Button(self._frame, width=39, pady=7, text="Log-In", bg="#57a1f8",
               fg="white", border=0, command=self.log_in).place(x=35, y=254)

        self._label = Label(self._frame, text="Don't have an account?", fg="black",
                            bg="white", font=("Microsoft YaHei UI Light", 9))
        self._label.place(x=75, y=320)

        # Move to sign up page button
        self._sign_up = Button(self._frame, width=6, text="Sign up", border=0, bg="white",
                               cursor="hand2", fg="#57a1f8", command=self.move_to_signup)
        self._sign_up.place(x=215, y=320)

        self._root.mainloop()

    def hide_password(self,):
        self._eye_picture.config(file='close_eye.png')
        self._code.config(show='*')
        self._eye_button.config(command=self.show_password)

    def show_password(self,):
        self._eye_picture.config(file='open_eye.png')
        self._code.config(show='')
        self._eye_button .config(command=self.hide_password)

    def log_in(self,):
        username = self._user.get()
        password = self._code.get()

        client.send(f"{username}${password}".encode())
        message = client.recv(1024).decode()

        if message == "LOGIN":
            self.move_to_main_page()
        elif message == "WRONG":
            messagebox.showerror("Error", "wrong username or password")  # wrong password
        elif message == "ERROR":
            messagebox.showerror("Error", "wrong username or password")  # wrong username


class SignUpPage(Window):
    def __init__(self,):
        # define the secondary frame
        Window.__init__(self, "925x500")

        # picture
        self._img = PhotoImage(file="login.png")
        Label(self._root, image=self._img, bg="white").place(x=50, y=90)

        # define the secondary frame
        self._frame = Frame(self._root, width=350, height=390, bg="white")
        self._frame.place(x=480, y=50)

        self._heading = Label(self._frame, text="Sign up", fg="#57a1f8",
                              bg="white", font=("Microsoft YaHei UI Light", 23, "bold"))
        self._heading.place(x=100, y=5)

        # receiving username block
        self._user = Entry(self._frame, width=25, fg="black", border=0,
                           bg="white", font=("Microsoft YaHei UI Light", 11))
        self._user.place(x=30, y=80)
        self._user.insert(0, "Username")
        self._user.bind("<FocusIn>", self.on_enter_username)
        self._user.bind("<FocusOut>", self.on_leave_username)
        Frame(self._frame, width=295, height=2, bg="black").place(x=25, y=107)

        # receiving password block
        self._code = Entry(self._frame, width=25, bg="white", fg="black",
                           border=0, font=("Microsoft YaHei UI Light", 11))
        self._code.place(x=30, y=150)
        self._code.insert(0, "Password")
        self._code.bind("<FocusIn>", self.on_enter_password)
        self._code.bind("<FocusOut>", self.on_leave_password)
        Frame(self._frame, width=295, height=2, bg="black").place(x=25, y=177)

        # receiving confirm password block
        self._confirm_pass = Entry(self._frame, width=25, bg="white", fg="black",
                                   border=0, font=("Microsoft YaHei UI Light", 11))
        self._confirm_pass.place(x=30, y=220)
        self._confirm_pass.insert(0, "Confirm password")
        self._confirm_pass.bind("<FocusIn>", self.on_enter_confirm_password)
        self._confirm_pass.bind("<FocusOut>", self.on_leave_confirm_password)
        Frame(self._frame, width=295, height=2, bg="black").place(x=25, y=247)

        # Move to site page button
        Button(self._frame, width=39, pady=7, text="Create", bg="#57a1f8",
               fg="white", border=0, command=self.signup).place(x=35, y=280)

        Label(self._frame, text="I have an account", bg="white",
              fg="black", font=("Microsoft YaHei UI Light", 9)).place(x=90, y=340)

        # Move to login page button
        Button(self._frame, width=6, text="Log in", border=0, bg="white",
               fg="#57a1f8", cursor="hand2", command=self.move_to_login).place(x=200, y=340)

        self._root.mainloop()

    def on_enter_username(self, e):
        name = self._user.get()
        if name == "Username":
            self._user.delete(0, "end")

    def on_leave_username(self, e):
        name = self._user.get()
        if name == "":
            self._user.insert(0, "Username")

    def on_enter_password(self, e):
        name = self._code.get()
        if name == "Password":
            self._code.delete(0, "end")

    def on_leave_password(self, e):
        name = self._code.get()
        if name == "":
            self._code.insert(0, "Password")

    def on_enter_confirm_password(self, e):
        name = self._confirm_pass.get()
        if name == "Confirm password":
            self._confirm_pass.delete(0, "end")

    def on_leave_confirm_password(self, e):
        name = self._confirm_pass.get()
        if name == "":
            self._confirm_pass.insert(0, "Confirm password")

    def signup(self,):
        username = self._user.get()
        password = self._code.get()
        confirmed_pass = self._confirm_pass.get()
        flag = False    # contain special char flag

        for letter in string.punctuation:
            if letter in username or letter in password:
                flag = True

        if flag:
            # special characters = !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~
            messagebox.showerror("Invalid Characters", f"Can't use '{string.punctuation}' in username or password")
        elif len(username) < 4:
            messagebox.showerror("Invalid Username Length",
                                 "Username is too short, username needs to be at least 4 letters")
        elif len(password) < 8:
            messagebox.showerror("Invalid Password Length",
                                 "Password is too short, password needs to be at least 8 letters")
        elif password.isnumeric():
            messagebox.showerror("Invalid Password characters", "password must contain numbers")
        elif password != confirmed_pass:
            messagebox.showerror("Error", "The confirmed password and the original password do not match")
        else:	  # send info to the server
            client.send(f"{username}@{password}".encode())
            message = client.recv(1024).decode()
            if message == "ERROR":
                messagebox.showerror("Error!", "Username is already exist")
            else:
                messagebox.showinfo("Succeeded!", "User was created successfully")
                self.move_to_home()


class MainPage(Window):
    def __init__(self,):
        # define the main frame
        Window.__init__(self, "400x500")
        self._root.configure(bg="grey51")
        # define the secondary frame
        self._frame = Frame(self._root, width=350, height=480, bg="grey")
        self._frame.place(x=25, y=10)

        self._heading = Label(self._frame, text="Enter The Company's Name", fg="black",
                              bg="grey", font=("Microsoft YaHei UI Light", 15, "bold"))
        self._heading.place(x=30, y=30)

        # Receiving company's name
        self._company = Entry(self._frame, width=15, fg="black", border=1,
                              bg="lightgrey", justify=CENTER, font=("Microsoft YaHei UI Light", 11))
        self._company.place(x=105, y=80)

        # send name of company to server button
        # Button(self._frame, width=20, pady=2, text="Forecast Using RNN", bg="black",
        #        fg="white", border=0, command=lambda: self.send_name("rnn")).place(x=94, y=135)

        Button(self._frame, width=20, pady=2, text="Forecast Price", bg="black",
               fg="white", border=0, command=lambda: self.send_name("lstm")).place(x=94, y=165)

        # Move to sign up page button
        self._label = Label(self._frame, text="Move to home page - ", fg="black",
                            bg="grey", font=("Microsoft YaHei UI Light", 9))
        self._label.place(x=80, y=320)

        self._sign_up = Button(self._frame, width=6, text="Home", border=0, bg="grey",
                               cursor="hand2", fg="#57a1f8", command=self.move_to_home)
        self._sign_up.place(x=208, y=320)
        self._root.mainloop()

    def send_name(self, method):
        company = self._company.get().upper()
        client.send(f"{company}!{method}".encode())

        message = client.recv(1024).decode()
        if message not in ["Higher", "Lower"]:
            messagebox.showerror("ERROR", "the company you mentioned doesn't exists in the database")
        else:
            messagebox.showinfo("Status", f"{company}'s stock price will get {message}")


if __name__ == "__main__":
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect((HOST, PORT))
    HomePage()
