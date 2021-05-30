from utils.data_structure import BufferedLazyList

render = True

boss = None
player = None
bullets = BufferedLazyList()
weapons = BufferedLazyList()
others = BufferedLazyList()
# player_bullets = BufferedLazyList()


def update():
    global boss, player, bullets, weapons, others
    boss.update()
    player.update()
    bullets.update(lambda x:x.dead)
    weapons.update()
    others.update(lambda x:x.dead)

def clear():
    global boss, player, bullets, weapons, others
    boss = None
    player = None
    bullets.clear()
    weapons.clear()
    others.clear()
    # player_bullets = BufferedLazyList()
    # player_bullets.update()
