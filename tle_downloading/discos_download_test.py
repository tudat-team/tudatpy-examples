from tudatpy.data.discos import DiscosQuery

token = 'ImU4NjY4OTBkLTVmOGYtNGIzNC1hZDg2LTRiYzVjZGYzMGJmMSI.JFqNE3ULKz5JE8WaIBmDGvS0V4o' # personal access token for Luigi Gisolfi
dq = DiscosQuery(token)
# Query object 40485 and print attributes
attributes = dq.query_object(10000)
mass = dq.get_object_attribute(10000, 'masas')
print(mass)