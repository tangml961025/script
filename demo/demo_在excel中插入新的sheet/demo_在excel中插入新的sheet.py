import pandas as pd
from openpyxl import load_workbook

df = pd.read_excel('./test2.xlsx')
book = load_workbook('test2.xlsx')
writer = pd.ExcelWriter('test2.xlsx', engine='openpyxl')
writer.book = book
writer.sheets = dict((ws.title, ws) for ws in book.worksheets)
df.to_excel(writer, "0620",index=0,startrow=0,startcol=0)
writer.save()