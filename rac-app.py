import streamlit as st
import about
import Home3
import rca
import rca2

from multiapp import MultiApp

app = MultiApp()

# Add all your application here
app.add_app("Home", Home3.app)
app.add_app("Failure mode classification app", rca.app)
app.add_app("Capacity prediction app", rca2.app)
app.add_app("About", about.app)


# The main app
app.run()
