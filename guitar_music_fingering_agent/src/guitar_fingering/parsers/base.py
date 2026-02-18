"""
Classe de base abstraite pour les parsers de fichiers musicaux.
"""

from abc import ABC, abstractmethod

from guitar_fingering.models.music import Track


class BaseParser(ABC):
    """Classe de base pour tous les parsers de fichiers musicaux.

    Chaque parser doit implémenter la méthode `parse()` qui retourne
    un objet Track contenant les données musicales extraites.
    """

    @abstractmethod
    def parse(self, filepath: str) -> Track:
        """Parse un fichier musical et retourne une piste guitare.

        Args:
            filepath: Chemin vers le fichier musical.

        Returns:
            Objet Track contenant les notes et mesures extraites.
        """
        ...

    @abstractmethod
    def supported_extensions(self) -> list[str]:
        """Retourne la liste des extensions de fichier supportées.

        Returns:
            Liste d'extensions (ex: ['.gp5', '.gp4', '.gp3']).
        """
        ...
