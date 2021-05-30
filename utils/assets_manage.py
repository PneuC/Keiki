from pygame import image
from root import PRJROOT


class TexManager:
    __instance = None
    __bullet_path = PRJROOT + 'assets/bullets/'
    __player_path = PRJROOT + 'assets/players/'
    __boss_path = PRJROOT + 'assets/boss/HoujuuBall.png'

    def __init__(self):
        print('tex_mgnr init')
        self.bullet_sheets = TexManager.__load_bullet_sheets()
        self.player_body, self.hitboxes  = TexManager.__load_player_related()
        self.main, self.sub0, self.sub1, self.sub = TexManager.__load_weapon_related()
        self.boss_tex = TexManager.__load_boss_related()
        TexManager.__instance = self

    def get_bullet_tex(self, btype, color):
        return self.bullet_sheets[btype.value][color.value]

    @staticmethod
    def inst():
        if TexManager.__instance is None:
            TexManager()
        return TexManager.__instance

    @staticmethod
    def __load_player_related():
        body_image = image.load(TexManager.__player_path + 'body.png')
        body_sheet = tuple(
            tuple(body_image.subsurface(j * 32, i * 48, 32, 48) for j in range(8))
            for i in range(3)
        )
        hitboxes = (
            image.load(TexManager.__player_path + 'hitbox0.png'),
            image.load(TexManager.__player_path + 'hitbox1.png')
        )
        return body_sheet, hitboxes

    @staticmethod
    def __load_weapon_related():
        main = image.load(TexManager.__player_path + 'main.png')
        sub0 = image.load(TexManager.__player_path + 'sub0.png')
        sub1 = image.load(TexManager.__player_path + 'sub1.png')
        sub = image.load(TexManager.__player_path + 'sub.png')
        return main, sub0, sub1, sub

    @staticmethod
    def __load_boss_related():
        return image.load(TexManager.__boss_path)

    @staticmethod
    def __load_bullet_sheets():
        tiny_sheet = image.load(TexManager.__bullet_path + 'tiny.png')
        mid_sheet = image.load(TexManager.__bullet_path + 'mid.png')
        big_sheet = image.load(TexManager.__bullet_path + 'large.png')
        giant_sheet = image.load(TexManager.__bullet_path + 'giant.png')
        sh = [tuple(tiny_sheet.subsurface(j * 8, 0, 8, 8) for j in range(6))]
        sh += [
            tuple(mid_sheet.subsurface(j * 16, i * 16, 16, 16) for j in range(6))
            for i in range(9)
        ]
        sh += [
            tuple(big_sheet.subsurface(j * 32, i * 32, 32, 32) for j in range(6))
            for i in range(4)
        ]
        sh += [tuple(giant_sheet.subsurface(j * 64, 0, 64, 64) for j in range(6))]
        return tuple(sh)

