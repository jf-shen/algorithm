import numpy as np 
import datetime

class Global:
	def __init__(self):
		self.days = 0

class Context:
	def __init__(self):
		self.current_dt = datetime.datetime.now()

g = Global()
context = Context()

