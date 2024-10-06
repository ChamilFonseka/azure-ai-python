from dotenv import load_dotenv
import os
from azure.core.credentials import AzureKeyCredential
from azure.keyvault.secrets import SecretClient
from azure.identity import ClientSecretCredential
from azure.ai.textanalytics import (
    TextAnalyticsClient,
    RecognizeEntitiesAction,
    RecognizeLinkedEntitiesAction,
    RecognizePiiEntitiesAction,
    ExtractKeyPhrasesAction,
    AnalyzeSentimentAction,
)

def main():
    global ai_endpoint
    global cog_key

    try:
        # Get Configuration Settings
        load_dotenv()
        ai_endpoint = os.getenv('AI_SERVICE_ENDPOINT')        
        key_vault_name = os.getenv('KEY_VAULT')        
        app_tenant = os.getenv('TENANT_ID')        
        app_id = os.getenv('APP_ID')        
        app_password = os.getenv('APP_PASSWORD')

        # Get Azure AI services key from keyvault using the service principal credentials        
        key_vault_uri = f"https://{key_vault_name}.vault.azure.net/"        
        credential = ClientSecretCredential(app_tenant, app_id, app_password)        
        keyvault_client = SecretClient(key_vault_uri, credential)        
        secret_key = keyvault_client.get_secret("AI-Services-Key")        
        cog_key = secret_key.value
    
        client = TextAnalyticsClient(endpoint=ai_endpoint, credential=AzureKeyCredential(cog_key))

        documents = [
            'We went to Contoso Steakhouse located at midtown NYC last week for a dinner party, and we adore the spot! '
            'They provide marvelous food and they have a great menu. The chief cook happens to be the owner (I think his name is John Doe) '
            'and he is super nice, coming out of the kitchen and greeted us all.'
        ]

        poller = client.begin_analyze_actions(
            documents,
            display_name="Sample Text Analysis",
            actions=[
                RecognizeEntitiesAction(),
                RecognizePiiEntitiesAction(),
                ExtractKeyPhrasesAction(),
                RecognizeLinkedEntitiesAction(),
                AnalyzeSentimentAction(),
            ],
        )

        document_results = poller.result()
        for doc, action_results in zip(documents, document_results):
            print(f"\nDocument text: {doc}")
            for result in action_results:
                if result.kind == "EntityRecognition":
                    print("...Results of Recognize Entities Action:")
                    for entity in result.entities:
                        print(f"......Entity: {entity.text}")
                        print(f".........Category: {entity.category}")
                        print(f".........Confidence Score: {entity.confidence_score}")
                        print(f".........Offset: {entity.offset}")

                elif result.kind == "PiiEntityRecognition":
                    print("...Results of Recognize PII Entities action:")
                    for pii_entity in result.entities:
                        print(f"......Entity: {pii_entity.text}")
                        print(f".........Category: {pii_entity.category}")
                        print(f".........Confidence Score: {pii_entity.confidence_score}")

                elif result.kind == "KeyPhraseExtraction":
                    print("...Results of Extract Key Phrases action:")
                    print(f"......Key Phrases: {result.key_phrases}")

                elif result.kind == "EntityLinking":
                    print("...Results of Recognize Linked Entities action:")
                    for linked_entity in result.entities:
                        print(f"......Entity name: {linked_entity.name}")
                        print(f".........Data source: {linked_entity.data_source}")
                        print(f".........Data source language: {linked_entity.language}")
                        print(
                            f".........Data source entity ID: {linked_entity.data_source_entity_id}"
                        )
                        print(f".........Data source URL: {linked_entity.url}")
                        print(".........Document matches:")
                        for match in linked_entity.matches:
                            print(f"............Match text: {match.text}")
                            print(f"............Confidence Score: {match.confidence_score}")
                            print(f"............Offset: {match.offset}")
                            print(f"............Length: {match.length}")

                elif result.kind == "SentimentAnalysis":
                    print("...Results of Analyze Sentiment action:")
                    print(f"......Overall sentiment: {result.sentiment}")
                    print(
                        f"......Scores: positive={result.confidence_scores.positive}; \
                        neutral={result.confidence_scores.neutral}; \
                        negative={result.confidence_scores.negative} \n"
                    )

                elif result.is_error is True:
                    print(
                        f"...Is an error with code '{result.error.code}' and message '{result.error.message}'"
                    )
                    
            print("------------------------------------------")        

    except Exception as ex:
        print(ex)

if __name__ == "__main__":
    main()