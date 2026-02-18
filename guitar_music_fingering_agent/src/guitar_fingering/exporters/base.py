"""
Classe de base abstraite pour les exporteurs de fichiers musicaux annotés.
"""

from abc import ABC, abstractmethod

from guitar_fingering.models.music import Track


class BaseExporter(ABC):
    """Classe de base pour tous les exporteurs.

    Chaque exporteur doit implémenter la méthode `export()` qui écrit
    un fichier musical avec les annotations de doigté.
    """

    @abstractmethod
    def export(self, track: Track, filepath: str) -> None:
        """Exporte une piste avec doigté vers un fichier.

        Args:
            track: Objet Track avec doigté assigné.
            filepath: Chemin du fichier de sortie.
        """
        ...

    @abstractmethod
    def supported_extensions(self) -> list[str]:
        """Retourne la liste des extensions de fichier supportées pour l'export.

        Returns:
            Liste d'extensions (ex: ['.gp5']).
        """
        ...
