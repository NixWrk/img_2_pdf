"""Collect CustomTkinter assets without repository metadata files."""

from PyInstaller.utils.hooks import collect_data_files


datas = collect_data_files("customtkinter", excludes=["**/.DS_Store"])
