import socket
import threading

import sqlite3
import hashlib
import csv

from RNN_CLASS import RNN
from LSTM_CLASS import LSTMPrediction

HOST = "0.0.0.0"
PORT = 52744


def create_databases():
    con = sqlite3.connect("database.db")
    cursor = con.cursor()

    try:
        cursor.execute("""CREATE TABLE names_and_passwords (
                          username TEXT PRIMARY KEY NOT NULL,
                          password TEXT);""")
        con.commit()
    except:
        pass

    try:
        cursor.execute("""CREATE TABLE symbols (
                              Symbol TEXT,
                              Name TEXT);""")
        con.commit()

        with open('names_and_symbols.csv') as file_obj:
            reader_obj = csv.reader(file_obj)
            next(reader_obj, None)

            for row in reader_obj:
                con.execute("INSERT INTO symbols (Symbol, Name) VALUES (?, ?);", (row[0].upper(), row[1].upper()))
                con.commit()

    except:
        pass
    finally:
        con.close()


# sign up. Use "@"
def enter_info_into_database(message, client):
    client_user = message[0]
    client_pass = message[1]

    encryptor = hashlib.sha512()
    encryptor.update(client_user.encode("utf8"))
    hashed_username = encryptor.digest()
    encryptor.update(client_pass.encode("utf8"))
    hashed_password = encryptor.digest()

    con = sqlite3.connect("database.db")

    try:  # tries to add the new user (username must be unique)
        con.execute("INSERT INTO names_and_passwords (username, password) VALUES (?, ?);", (hashed_username, hashed_password))
        con.commit()
        con.close()
        client.send("CREATED".encode())
    except:  # username already exist
        con.close()
        client.send("ERROR".encode())


# log-in. Use "$"
def check_if_user_info_is_correct(message, client):
    client_user = message[0]
    client_pass = message[1]

    encryptor = hashlib.sha512()
    encryptor.update(client_user.encode("utf8"))
    hashed_username = encryptor.digest()
    encryptor.update(client_pass.encode("utf8"))
    hashed_password = encryptor.digest()

    con = sqlite3.connect("database.db")
    tmp = con.execute(f"SELECT * FROM names_and_passwords WHERE username = ?", (hashed_username, )).fetchall()
    con.close()

    try:    # username exists
        database_pass = tmp[0][1]

        if hashed_password == database_pass:    # password match the username
            client.send("LOGIN".encode())
            print("login")
        else:                               # password does not match the username
            client.send("WRONG".encode())
            print("wrong")
    except:     # username doesnt exist
        client.send("ERROR".encode())


# search symbol name
def check_symbol_exists(company_name):
    company_name = company_name.upper()
    con = sqlite3.connect("database.db")
    cursor = con.cursor()

    company_name = cursor.execute(f"SELECT Symbol FROM symbols WHERE Symbol = ?", (company_name,)).fetchall()
    con.close()

    if len(company_name) == 0:
        return "None"
    else:
        return company_name[0][0]


# model. Use "!"
def predict(symbol, method, client):
    if symbol == "None":
        client.send("ERROR".encode())
    else:
        if method == "rnn":
            model = RNN(symbol)
            client.send(model.value.encode())
        elif method == "lstm":
            model = LSTMPrediction(symbol)
            client.send(model.value.encode())


def client_handler(client):
    while True:
        message = client.recv(1024).decode()    # [username, password] or the info to the project page
        if "$" in message:  # login case
            message = message.split("$")
            check_if_user_info_is_correct(message, client)
        elif "@" in message:               # sign-up case
            message = message.split("@")
            enter_info_into_database(message, client)
        elif "!" in message:
            message = message.split("!")
            symbol = check_symbol_exists(message[0])
            method = message[1]
            predict(symbol, method, client)


def connection():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((HOST, PORT))
    server.listen()
    print("Listening...\n")

    while True:
        print("Waiting for a new client...")
        client_socket, address = server.accept()
        print("Client connected...\n")
        thread = threading.Thread(target=client_handler, args=(client_socket,))
        thread.start()


if __name__ == "__main__":
    connection()

