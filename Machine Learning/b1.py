def getNewBoard():
    # Creates a brand-new, blank board data structure.
    board = []
    for i in range(8):
        board.append([' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '])
    return board

print(getNewBoard())