import curses
import curses.textpad


window = curses.initscr()
window.keypad(True)

def get_input_ch():
    """ curses.textpad module is needed """
    while True:
        ch = window.getch()
        if ch == curses.KEY_UP:
            return "1__Scroll(UP)"
        elif ch == curses.KEY_DOWN:
            return "2__Scroll(DOWN)"
        elif ch == curses.KEY_LEFT:
            return "3__Paging(UP)"
        elif ch == curses.KEY_RIGHT:
            return "4__Paging(DOWN)"
        elif ch == curses.ascii.ESC:
            break
        else:
            continue
    return "5__ESC breaking"

while 1:
    print(get_input_ch())
