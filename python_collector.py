# STRING
flowers = "pink primrose,hard-leaved pocket orchid,canterbury bells,sweet pea,english marigold,tiger lily,moon orchid,bird of paradise,monkshood,globe thistle"


# LIST
flowers_list = ["pink primrose", "hard-leaved pocket orchid", "canterbury bells", "sweet pea", "english marigold", "tiger lily", "moon orchid", "bird of paradise", "monkshood", "globe thistle"]
print("First entry:", flowers_list[0])

# pulling only some elements of the list
print("First three entries:", flowers_list[:3])
print("Final two entries:", flowers_list[-2:])

# remove items
flowers_list.remove("globe thistle")

#add items
flowers_list.append("snapdragon")
mylist1.index(min(mylist1)) #get minimum of a list and its position in the list

print("Validation MAE: {:,.0f}".format(val_mae))
print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_mae))

# formattazione di numero con la virgola
print("Min mae: {:.2f}".format(min(mae_list))) --> 27282.50803885739->27282.51
