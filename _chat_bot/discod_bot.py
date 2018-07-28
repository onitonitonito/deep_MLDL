import random
import discord
from discord.ext import commands
from discord.voice_client import VoiceClient


startup_extensions = ["Music",]
bot = commands.Bot("!")
client = discord.Client()

@bot.event
async def on_ready():
    print("...  정상작동중  ...")


class MainCommands(object):
    def __init__(self, bot):
        self.bot = bot


if __name__ == '__main__':
    for extension in startup_extensions:
        try:
            bot.load_extension(extension)

        except Exception as e:
            exc = "{}: {}".format(type(e).__name__, e)
            print("Failed to load extension {} \n{}").format(extension, exc)


bot.run("{TOKEN_HASH_PRINT_UNKNOWN_HERE}")
a = input()



""" pip install discord
Successfully built discord discord.py aiohttp
web3 4.7.2 has requirement websockets<7.0.0,>=6.0.0,
but you'll have websockets 3.4 which is incompatible.
"""
