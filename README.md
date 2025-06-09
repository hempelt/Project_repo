# Project_repo

Regressionsmodell zur Vorhersage des Schmelzpunktes von Antikörpern in unterschiedlichen Formulierungen an Hand der verwendeten Hilfsstoffe und Moleküleigenschaften.
 
Beschreibung
 
Der Schmelzpunkt ist ein Maß für die Stabilität von Proteinen und ist abhängig von den Eigenschaften des Proteins, der Proteinkonzentration und den verwendeten Hilfsstoffen, die der Lösung zugesetzt wurden. Anhand mehrerer Prädiktoren soll der Schmelzpunkt quantitativ vorhergesagt und das Model validiert werden.
 
Datensatz:
 
Daten wurden von Labormitarbeitern über die letzten 2 Jahre gemessen und in einer Datenbank gespeichert. Kleiner Ausschnitt aus dem Datenbestand (Daten müssen noch bereinigt und transformiert werden). ca. 400000 Datensätze

## Installation

```bash
python -m venv project_env
.\project_env\Scripts\Activate.ps1
pip install -r requirements.txt