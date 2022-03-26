import streamlit as st
from multiapp import MultiApp
import Home3
import rca
import ndf2
import rca2
import cores
import about
#import rca2

app = MultiApp()

# Add all your application here
app.add_app("Home", Home3.app)
app.add_app("Failure mode classification app", rca.app)
app.add_app("Capacity prediction app", rca2.app)
app.add_app("About", about.app)


# The main app
app.run()
