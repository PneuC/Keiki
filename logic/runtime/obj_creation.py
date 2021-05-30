from utils.math import Vec2
from utils.assets_manage import TexManager
from logic.objects.player import Player
from logic.objects.boss import Boss
from logic.objects.weapons import Sub, Weapon
from logic.runtime import objs
from render.real_time.player_sprites import PlayerBody, PlayerHitBox
from render.real_time.renderer import FullRenderer
from render.real_time.boss_sprites import HoujuuBallSprites
from render.real_time.common_sprites import CommonSprite


def create_player(render=True):
    objs.player = Player()
    objs.weapons.add(Weapon(objs.player, 'main', 3, 10.8, (Vec2(-7, 32), Vec2(7, 32))))
    subs = [
        Sub(objs.player, (Vec2(-36, -12), Vec2(-16, 32))),
        Sub(objs.player, (Vec2(36, -12), Vec2(16, 32))),
        Sub(objs.player, (Vec2(-18, -30), Vec2(-32, 20))),
        Sub(objs.player, (Vec2(18, -30), Vec2(32, 20)))
    ]
    objs.others.add(*subs)
    objs.weapons.add(
        *(Weapon(sub, 'sub1', 5, 9.0, [Vec2(0, 30)]) for sub in subs)
    )
    if render:
        FullRenderer.instance.add_sprites(
            FullRenderer.Layers.Player,
            PlayerBody(objs.player)
        )
        FullRenderer.instance.add_sprites(
            FullRenderer.Layers.Player,
            *(CommonSprite(sub, TexManager.inst().sub, spinable=True) for sub in subs)
        )
        FullRenderer.instance.add_sprites(
            FullRenderer.Layers.PlayerHitBox,
            PlayerHitBox(objs.player, 0), PlayerHitBox(objs.player, 1)
        )


def create_boss(render=True):
    objs.boss = Boss()
    if render:
        FullRenderer.instance.add_sprites(
            FullRenderer.Layers.Player, HoujuuBallSprites(objs.boss)
        )


