"""
This file contains the main interface and the public API for multilspy. 
The abstract class LanguageServer provides a factory method, creator that is 
intended for creating instantiations of language specific clients.
The details of Language Specific configuration are not exposed to the user.
"""

import asyncio
from copy import copy
import dataclasses
import hashlib
import json
import pickle
import time
import logging
import os
import pathlib
import threading
from contextlib import asynccontextmanager, contextmanager
from .lsp_protocol_handler.lsp_constants import LSPConstants
from  .lsp_protocol_handler import lsp_types as LSPTypes

from . import multilspy_types
from .multilspy_logger import MultilspyLogger
from .lsp_protocol_handler.server import (
    LanguageServerHandler,
    ProcessLaunchInfo,
)
from .multilspy_config import MultilspyConfig, Language
from .multilspy_exceptions import MultilspyException
from .multilspy_utils import PathUtils, FileUtils, TextUtils
from pathlib import Path, PurePath
from typing import AsyncIterator, Iterator, List, Dict, Optional, Union, Tuple
from .type_helpers import ensure_all_methods_implemented


@dataclasses.dataclass
class LSPFileBuffer:
    """
    This class is used to store the contents of an open LSP file in memory.
    """

    # uri of the file
    uri: str

    # The contents of the file
    contents: str

    # The version of the file
    version: int

    # The language id of the file
    language_id: str

    # reference count of the file
    ref_count: int
    
    # --------------------------------- MODIFICATIONS BY MISCHA ---------------------------------
    
    content_hash: str = ""
    
    def __post_init__(self):
        self.content_hash = hashlib.md5(self.contents.encode('utf-8')).hexdigest()


class LanguageServer:
    """
    The LanguageServer class provides a language agnostic interface to the Language Server Protocol.
    It is used to communicate with Language Servers of different programming languages.
    """

    @classmethod
    def create(cls, config: MultilspyConfig, logger: MultilspyLogger, repository_root_path: str) -> "LanguageServer":
        """
        Creates a language specific LanguageServer instance based on the given configuration, and appropriate settings for the programming language.

        If language is Java, then ensure that jdk-17.0.6 or higher is installed, `java` is in PATH, and JAVA_HOME is set to the installation directory.
        If language is JS/TS, then ensure that node (v18.16.0 or higher) is installed and in PATH.

        :param repository_root_path: The root path of the repository.
        :param config: The Multilspy configuration.
        :param logger: The logger to use.

        :return LanguageServer: A language specific LanguageServer instance.
        """
        if config.code_language == Language.PYTHON:
            from multilspy.language_servers.pyright_language_server.pyright_server import (
                PyrightServer,
            )

            return PyrightServer(config, logger, repository_root_path)
            # It used to be jedi, but pyright is a bit faster, and also more actively maintained
            # Keeping the previous code for reference
            # from multilspy.language_servers.jedi_language_server.jedi_server import (
            #     JediServer,
            # )

            # return JediServer(config, logger, repository_root_path)
        elif config.code_language == Language.JAVA:
            from multilspy.language_servers.eclipse_jdtls.eclipse_jdtls import (
                EclipseJDTLS,
            )

            return EclipseJDTLS(config, logger, repository_root_path)
        elif config.code_language == Language.RUST:
            from multilspy.language_servers.rust_analyzer.rust_analyzer import (
                RustAnalyzer,
            )

            return RustAnalyzer(config, logger, repository_root_path)
        elif config.code_language == Language.CSHARP:
            from multilspy.language_servers.omnisharp.omnisharp import OmniSharp

            return OmniSharp(config, logger, repository_root_path)
        elif config.code_language in [Language.TYPESCRIPT, Language.JAVASCRIPT]:
            from multilspy.language_servers.typescript_language_server.typescript_language_server import (
                TypeScriptLanguageServer,
            )
            return TypeScriptLanguageServer(config, logger, repository_root_path)
        elif config.code_language == Language.GO:
            from multilspy.language_servers.gopls.gopls import Gopls

            return Gopls(config, logger, repository_root_path)
        elif config.code_language == Language.RUBY:
            from multilspy.language_servers.solargraph.solargraph import Solargraph

            return Solargraph(config, logger, repository_root_path)
        else:
            logger.log(f"Language {config.code_language} is not supported", logging.ERROR)
            raise MultilspyException(f"Language {config.code_language} is not supported")

    def __init__(
        self,
        config: MultilspyConfig,
        logger: MultilspyLogger,
        repository_root_path: str,
        process_launch_info: ProcessLaunchInfo,
        language_id: str,
    ):
        """
        Initializes a LanguageServer instance.

        Do not instantiate this class directly. Use `LanguageServer.create` method instead.

        :param config: The Multilspy configuration.
        :param logger: The logger to use.
        :param repository_root_path: The root path of the repository.
        :param cmd: Each language server has a specific command used to start the server.
                    This parameter is the command to launch the language server process.
                    The command must pass appropriate flags to the binary, so that it runs in the stdio mode,
                    as opposed to HTTP, TCP modes supported by some language servers.
        """
        if type(self) == LanguageServer:
            raise MultilspyException(
                "LanguageServer is an abstract class and cannot be instantiated directly. Use LanguageServer.create method instead."
            )

        self.logger = logger
        self.server_started = False
        self.repository_root_path: str = repository_root_path
        self.completions_available = asyncio.Event()

        if config.trace_lsp_communication:

            def logging_fn(source, target, msg):
                self.logger.log(f"LSP: {source} -> {target}: {str(msg)}", logging.DEBUG)

        else:

            def logging_fn(source, target, msg):
                pass

        # cmd is obtained from the child classes, which provide the language specific command to start the language server
        # LanguageServerHandler provides the functionality to start the language server and communicate with it
        self.server: LanguageServerHandler = LanguageServerHandler(process_launch_info, logger=logging_fn)

        self.language_id = language_id
        self.open_file_buffers: Dict[str, LSPFileBuffer] = {}
        
        # --------------------------------- MODIFICATIONS BY MISCHA ---------------------------------
        self._document_symbols_cache:  dict[str, Tuple[str, Tuple[List[multilspy_types.UnifiedSymbolInformation], Optional[List[multilspy_types.TreeRepr]]]]] = {}
        """Maps file paths to a tuple of (file_content_hash, result_of_request_document_symbols)"""
        self.load_cache()
        self._cache_has_changed = bool

    @asynccontextmanager
    async def start_server(self) -> AsyncIterator["LanguageServer"]:
        """
        Starts the Language Server and yields the LanguageServer instance.

        Usage:
        ```
        async with lsp.start_server():
            # LanguageServer has been initialized and ready to serve requests
            await lsp.request_definition(...)
            await lsp.request_references(...)
            # Shutdown the LanguageServer on exit from scope
        # LanguageServer has been shutdown
        ```
        """
        self.server_started = True
        yield self
        self.server_started = False

    # TODO: Add support for more LSP features

    @contextmanager
    def open_file(self, relative_file_path: str) -> Iterator[LSPFileBuffer]:
        """
        Open a file in the Language Server. This is required before making any requests to the Language Server.

        :param relative_file_path: The relative path of the file to open.
        """
        if not self.server_started:
            self.logger.log(
                "open_file called before Language Server started",
                logging.ERROR,
            )
            raise MultilspyException("Language Server not started")

        absolute_file_path = str(PurePath(self.repository_root_path, relative_file_path))
        uri = pathlib.Path(absolute_file_path).as_uri()

        if uri in self.open_file_buffers:
            assert self.open_file_buffers[uri].uri == uri
            assert self.open_file_buffers[uri].ref_count >= 1

            self.open_file_buffers[uri].ref_count += 1
            yield self.open_file_buffers[uri]
            self.open_file_buffers[uri].ref_count -= 1
        else:
            contents = FileUtils.read_file(self.logger, absolute_file_path)

            version = 0
            self.open_file_buffers[uri] = LSPFileBuffer(uri, contents, version, self.language_id, 1)

            self.server.notify.did_open_text_document(
                {
                    LSPConstants.TEXT_DOCUMENT: {
                        LSPConstants.URI: uri,
                        LSPConstants.LANGUAGE_ID: self.language_id,
                        LSPConstants.VERSION: 0,
                        LSPConstants.TEXT: contents,
                    }
                }
            )
            yield self.open_file_buffers[uri]
            self.open_file_buffers[uri].ref_count -= 1

        if self.open_file_buffers[uri].ref_count == 0:
            self.server.notify.did_close_text_document(
                {
                    LSPConstants.TEXT_DOCUMENT: {
                        LSPConstants.URI: uri,
                    }
                }
            )
            del self.open_file_buffers[uri]

    def insert_text_at_position(
        self, relative_file_path: str, line: int, column: int, text_to_be_inserted: str
    ) -> multilspy_types.Position:
        """
        Insert text at the given line and column in the given file and return 
        the updated cursor position after inserting the text.

        :param relative_file_path: The relative path of the file to open.
        :param line: The line number at which text should be inserted.
        :param column: The column number at which text should be inserted.
        :param text_to_be_inserted: The text to insert.
        """
        if not self.server_started:
            self.logger.log(
                "insert_text_at_position called before Language Server started",
                logging.ERROR,
            )
            raise MultilspyException("Language Server not started")

        absolute_file_path = str(PurePath(self.repository_root_path, relative_file_path))
        uri = pathlib.Path(absolute_file_path).as_uri()

        # Ensure the file is open
        assert uri in self.open_file_buffers

        file_buffer = self.open_file_buffers[uri]
        file_buffer.version += 1
        change_index = TextUtils.get_index_from_line_col(file_buffer.contents, line, column)
        file_buffer.contents = (
            file_buffer.contents[:change_index] + text_to_be_inserted + file_buffer.contents[change_index:]
        )
        self.server.notify.did_change_text_document(
            {
                LSPConstants.TEXT_DOCUMENT: {
                    LSPConstants.VERSION: file_buffer.version,
                    LSPConstants.URI: file_buffer.uri,
                },
                LSPConstants.CONTENT_CHANGES: [
                    {
                        LSPConstants.RANGE: {
                            "start": {"line": line, "character": column},
                            "end": {"line": line, "character": column},
                        },
                        "text": text_to_be_inserted,
                    }
                ],
            }
        )
        new_l, new_c = TextUtils.get_updated_position_from_line_and_column_and_edit(line, column, text_to_be_inserted)
        return multilspy_types.Position(line=new_l, character=new_c)

    def delete_text_between_positions(
        self,
        relative_file_path: str,
        start: multilspy_types.Position,
        end: multilspy_types.Position,
    ) -> str:
        """
        Delete text between the given start and end positions in the given file and return the deleted text.
        """
        if not self.server_started:
            self.logger.log(
                "insert_text_at_position called before Language Server started",
                logging.ERROR,
            )
            raise MultilspyException("Language Server not started")

        absolute_file_path = str(PurePath(self.repository_root_path, relative_file_path))
        uri = pathlib.Path(absolute_file_path).as_uri()

        # Ensure the file is open
        assert uri in self.open_file_buffers

        file_buffer = self.open_file_buffers[uri]
        file_buffer.version += 1
        del_start_idx = TextUtils.get_index_from_line_col(file_buffer.contents, start["line"], start["character"])
        del_end_idx = TextUtils.get_index_from_line_col(file_buffer.contents, end["line"], end["character"])
        deleted_text = file_buffer.contents[del_start_idx:del_end_idx]
        file_buffer.contents = file_buffer.contents[:del_start_idx] + file_buffer.contents[del_end_idx:]
        self.server.notify.did_change_text_document(
            {
                LSPConstants.TEXT_DOCUMENT: {
                    LSPConstants.VERSION: file_buffer.version,
                    LSPConstants.URI: file_buffer.uri,
                },
                LSPConstants.CONTENT_CHANGES: [{LSPConstants.RANGE: {"start": start, "end": end}, "text": ""}],
            }
        )
        return deleted_text

    def get_open_file_text(self, relative_file_path: str) -> str:
        """
        Get the contents of the given opened file as per the Language Server.

        :param relative_file_path: The relative path of the file to open.
        """
        if not self.server_started:
            self.logger.log(
                "get_open_file_text called before Language Server started",
                logging.ERROR,
            )
            raise MultilspyException("Language Server not started")

        absolute_file_path = str(PurePath(self.repository_root_path, relative_file_path))
        uri = pathlib.Path(absolute_file_path).as_uri()

        # Ensure the file is open
        assert uri in self.open_file_buffers

        file_buffer = self.open_file_buffers[uri]
        return file_buffer.contents

    async def request_definition(
        self, relative_file_path: str, line: int, column: int
    ) -> List[multilspy_types.Location]:
        """
        Raise a [textDocument/definition](https://microsoft.github.io/language-server-protocol/specifications/lsp/3.17/specification/#textDocument_definition) request to the Language Server
        for the symbol at the given line and column in the given file. Wait for the response and return the result.

        :param relative_file_path: The relative path of the file that has the symbol for which definition should be looked up
        :param line: The line number of the symbol
        :param column: The column number of the symbol

        :return List[multilspy_types.Location]: A list of locations where the symbol is defined
        """

        if not self.server_started:
            self.logger.log(
                "find_function_definition called before Language Server started",
                logging.ERROR,
            )
            raise MultilspyException("Language Server not started")

        with self.open_file(relative_file_path):
            # sending request to the language server and waiting for response
            response = await self.server.send.definition(
                {
                    LSPConstants.TEXT_DOCUMENT: {
                        LSPConstants.URI: pathlib.Path(
                            str(PurePath(self.repository_root_path, relative_file_path))
                        ).as_uri()
                    },
                    LSPConstants.POSITION: {
                        LSPConstants.LINE: line,
                        LSPConstants.CHARACTER: column,
                    },
                }
            )

        ret: List[multilspy_types.Location] = []
        if isinstance(response, list):
            # response is either of type Location[] or LocationLink[]
            for item in response:
                assert isinstance(item, dict)
                if LSPConstants.URI in item and LSPConstants.RANGE in item:
                    new_item: multilspy_types.Location = {}
                    new_item.update(item)
                    new_item["absolutePath"] = PathUtils.uri_to_path(new_item["uri"])
                    try:
                        new_item["relativePath"] = str(
                            PurePath(os.path.relpath(new_item["absolutePath"], self.repository_root_path))
                        )
                    except:
                        new_item["relativePath"] = str(new_item["absolutePath"])
                    ret.append(multilspy_types.Location(new_item))
                elif (
                    LSPConstants.ORIGIN_SELECTION_RANGE in item
                    and LSPConstants.TARGET_URI in item
                    and LSPConstants.TARGET_RANGE in item
                    and LSPConstants.TARGET_SELECTION_RANGE in item
                ):
                    new_item: multilspy_types.Location = {}
                    new_item["uri"] = item[LSPConstants.TARGET_URI]
                    new_item["absolutePath"] = PathUtils.uri_to_path(new_item["uri"])
                    try:
                        new_item["relativePath"] = str(
                            PurePath(os.path.relpath(new_item["absolutePath"], self.repository_root_path))
                        )
                    except:
                        new_item["relativePath"] = str(new_item["absolutePath"])
                    new_item["range"] = item[LSPConstants.TARGET_SELECTION_RANGE]
                    ret.append(multilspy_types.Location(**new_item))
                else:
                    assert False, f"Unexpected response from Language Server: {item}"
        elif isinstance(response, dict):
            # response is of type Location
            assert LSPConstants.URI in response
            assert LSPConstants.RANGE in response

            new_item: multilspy_types.Location = {}
            new_item.update(response)
            new_item["absolutePath"] = PathUtils.uri_to_path(new_item["uri"])
            new_item["relativePath"] = str(
                PurePath(os.path.relpath(new_item["absolutePath"], self.repository_root_path))
            )
            ret.append(multilspy_types.Location(**new_item))
        elif response is None:
            # Some language servers return None when they cannot find a definition
            # This is expected for certain symbol types like generics or types with incomplete information
            self.logger.log(
                f"Language server returned None for definition request at {relative_file_path}:{line}:{column}",
                logging.WARNING,
            )
        else:
            assert False, f"Unexpected response from Language Server: {response}"

        return ret

    async def request_references(
        self, relative_file_path: str, line: int, column: int
    ) -> List[multilspy_types.Location]:
        """
        Raise a [textDocument/references](https://microsoft.github.io/language-server-protocol/specifications/lsp/3.17/specification/#textDocument_references) request to the Language Server
        to find references to the symbol at the given line and column in the given file. Wait for the response and return the result.

        :param relative_file_path: The relative path of the file that has the symbol for which references should be looked up
        :param line: The line number of the symbol
        :param column: The column number of the symbol

        :return List[multilspy_types.Location]: A list of locations where the symbol is referenced
        """

        if not self.server_started:
            self.logger.log(
                "find_all_callers_of_function called before Language Server started",
                logging.ERROR,
            )
            raise MultilspyException("Language Server not started")

        with self.open_file(relative_file_path):
            # sending request to the language server and waiting for response
            response = await self.server.send.references(
                {
                    "context": {"includeDeclaration": False},
                    "textDocument": {
                        "uri": pathlib.Path(os.path.join(self.repository_root_path, relative_file_path)).as_uri()
                    },
                    "position": {"line": line, "character": column},
                }
            )

        ret: List[multilspy_types.Location] = []
        assert isinstance(response, list)
        for item in response:
            assert isinstance(item, dict)
            assert LSPConstants.URI in item
            assert LSPConstants.RANGE in item

            new_item: multilspy_types.Location = {}
            new_item.update(item)
            new_item["absolutePath"] = PathUtils.uri_to_path(new_item["uri"])
            new_item["relativePath"] = str(
                PurePath(os.path.relpath(new_item["absolutePath"], self.repository_root_path))
            )
            ret.append(multilspy_types.Location(**new_item))

        return ret

    async def request_completions(
        self, relative_file_path: str, line: int, column: int, allow_incomplete: bool = False
    ) -> List[multilspy_types.CompletionItem]:
        """
        Raise a [textDocument/completion](https://microsoft.github.io/language-server-protocol/specifications/lsp/3.17/specification/#textDocument_completion) request to the Language Server
        to find completions at the given line and column in the given file. Wait for the response and return the result.

        :param relative_file_path: The relative path of the file that has the symbol for which completions should be looked up
        :param line: The line number of the symbol
        :param column: The column number of the symbol

        :return List[multilspy_types.CompletionItem]: A list of completions
        """
        with self.open_file(relative_file_path):
            open_file_buffer = self.open_file_buffers[
                pathlib.Path(os.path.join(self.repository_root_path, relative_file_path)).as_uri()
            ]
            completion_params: LSPTypes.CompletionParams = {
                "position": {"line": line, "character": column},
                "textDocument": {"uri": open_file_buffer.uri},
                "context": {"triggerKind": LSPTypes.CompletionTriggerKind.Invoked},
            }
            response: Union[List[LSPTypes.CompletionItem], LSPTypes.CompletionList, None] = None

            num_retries = 0
            while response is None or (response["isIncomplete"] and num_retries < 30):
                await self.completions_available.wait()
                response: Union[
                    List[LSPTypes.CompletionItem], LSPTypes.CompletionList, None
                ] = await self.server.send.completion(completion_params)
                if isinstance(response, list):
                    response = {"items": response, "isIncomplete": False}
                num_retries += 1

            # TODO: Understand how to appropriately handle `isIncomplete`
            if response is None or (response["isIncomplete"] and not(allow_incomplete)):
                return []

            if "items" in response:
                response = response["items"]

            response: List[LSPTypes.CompletionItem] = response

            # TODO: Handle the case when the completion is a keyword
            items = [item for item in response if item["kind"] != LSPTypes.CompletionItemKind.Keyword]

            completions_list: List[multilspy_types.CompletionItem] = []

            for item in items:
                assert "insertText" in item or "textEdit" in item
                assert "kind" in item
                completion_item = {}
                if "detail" in item:
                    completion_item["detail"] = item["detail"]
                
                if "label" in item:
                    completion_item["completionText"] = item["label"]
                    completion_item["kind"] = item["kind"]
                elif "insertText" in item:
                    completion_item["completionText"] = item["insertText"]
                    completion_item["kind"] = item["kind"]
                elif "textEdit" in item and "newText" in item["textEdit"]:
                    completion_item["completionText"] = item["textEdit"]["newText"]
                    completion_item["kind"] = item["kind"]
                elif "textEdit" in item and "range" in item["textEdit"]:
                    new_dot_lineno, new_dot_colno = (
                        completion_params["position"]["line"],
                        completion_params["position"]["character"],
                    )
                    assert all(
                        (
                            item["textEdit"]["range"]["start"]["line"] == new_dot_lineno,
                            item["textEdit"]["range"]["start"]["character"] == new_dot_colno,
                            item["textEdit"]["range"]["start"]["line"] == item["textEdit"]["range"]["end"]["line"],
                            item["textEdit"]["range"]["start"]["character"]
                            == item["textEdit"]["range"]["end"]["character"],
                        )
                    )
                    
                    completion_item["completionText"] = item["textEdit"]["newText"]
                    completion_item["kind"] = item["kind"]
                elif "textEdit" in item and "insert" in item["textEdit"]:
                    assert False
                else:
                    assert False

                completion_item = multilspy_types.CompletionItem(**completion_item)
                completions_list.append(completion_item)

            return [
                json.loads(json_repr)
                for json_repr in set([json.dumps(item, sort_keys=True) for item in completions_list])
            ]

    async def request_document_symbols(self, relative_file_path: str) -> Tuple[List[multilspy_types.UnifiedSymbolInformation], Union[List[multilspy_types.TreeRepr], None]]:
        """
        Raise a [textDocument/documentSymbol](https://microsoft.github.io/language-server-protocol/specifications/lsp/3.17/specification/#textDocument_documentSymbol) request to the Language Server
        to find symbols in the given file. Wait for the response and return the result.

        :param relative_file_path: The relative path of the file that has the symbols

        :return Tuple[List[multilspy_types.UnifiedSymbolInformation], Union[List[multilspy_types.TreeRepr], None]]: A list of symbols in the file, and the tree representation of the symbols
        """        
        self.logger.log(f"Requesting document symbols for {relative_file_path} for the first time", logging.DEBUG)
        with self.open_file(relative_file_path) as file_data:
            file_hash_and_result = self._document_symbols_cache.get(relative_file_path)
            if file_hash_and_result is not None:
                file_hash, result = file_hash_and_result
                if file_hash == file_data.content_hash:
                    self.logger.log(f"Returning cached document symbols for {relative_file_path}", logging.DEBUG)
                    return result
                else:
                    self.logger.log(f"Content for {relative_file_path} has changed. Overwriting cache", logging.INFO)
            
            
            response = await self.server.send.document_symbol(
                {
                    "textDocument": {
                        "uri": pathlib.Path(os.path.join(self.repository_root_path, relative_file_path)).as_uri()
                    }
                }
            )
        
        ret: List[multilspy_types.UnifiedSymbolInformation] = []
        l_tree = None
        assert isinstance(response, list)
        for item in response:
            assert isinstance(item, dict)
            assert LSPConstants.NAME in item
            assert LSPConstants.KIND in item

            if LSPConstants.CHILDREN in item:
                # TODO: l_tree should be a list of TreeRepr. Define the following function to return TreeRepr as well
                
                def visit_tree_nodes_and_build_tree_repr(tree: LSPTypes.DocumentSymbol) -> List[multilspy_types.UnifiedSymbolInformation]:
                    l: List[multilspy_types.UnifiedSymbolInformation] = []
                    children = tree['children'] if 'children' in tree else []
                    if 'children' in tree:
                        del tree['children']
                    l.append(multilspy_types.UnifiedSymbolInformation(**tree))
                    for child in children:
                        l.extend(visit_tree_nodes_and_build_tree_repr(child))
                    return l
                
                ret.extend(visit_tree_nodes_and_build_tree_repr(item))
            else:
                ret.append(multilspy_types.UnifiedSymbolInformation(**item))

        result = ret, l_tree
        self.logger.log(f"Caching document symbols for {relative_file_path}", logging.DEBUG)
        self._document_symbols_cache[relative_file_path] = (file_data.content_hash, result)
        self._cache_has_changed = True
        return result
    
    async def request_hover(self, relative_file_path: str, line: int, column: int) -> Union[multilspy_types.Hover, None]:
        """
        Raise a [textDocument/hover](https://microsoft.github.io/language-server-protocol/specifications/lsp/3.17/specification/#textDocument_hover) request to the Language Server
        to find the hover information at the given line and column in the given file. Wait for the response and return the result.

        :param relative_file_path: The relative path of the file that has the hover information
        :param line: The line number of the symbol
        :param column: The column number of the symbol

        :return None
        """
        with self.open_file(relative_file_path):
            response = await self.server.send.hover(
                {
                    "textDocument": {
                        "uri": pathlib.Path(os.path.join(self.repository_root_path, relative_file_path)).as_uri()
                    },
                    "position": {
                        "line": line,
                        "character": column,
                    },
                }
            )
        
        if response is None:
            return None

        assert isinstance(response, dict)

        return multilspy_types.Hover(**response)
    
    # ----------------------------- FROM HERE ON MODIFICATIONS BY MISCHA --------------------
    
    async def request_parsed_files(self) -> list[str]:
        """This is slow, as it finds all files by finding all symbols. 
        
        This seems to be the only way, the LSP does not provide any endpoints for listing project files."""
        if not self.server_started:
            self.logger.log(
                "request_parsed_files called before Language Server started",
                logging.ERROR,
            )
            raise MultilspyException("Language Server not started")
        
        params = LSPTypes.WorkspaceSymbolParams(query="")  # Empty query returns all symbols
        symbols = await self.server.send.workspace_symbol(params) or []
        return list({s["location"]["uri"].replace("file://", "") for s in symbols})
    
    async def request_referencing_symbols(
        self,
        relative_file_path: str,
        line: int,
        column: int,
        include_imports: bool = True,
        include_self: bool = False,
    ) -> List[multilspy_types.UnifiedSymbolInformation]:
        """
        Finds all symbols that reference the symbol at the given location.
        This is similar to request_references but filters to only include symbols
        (functions, methods, classes, etc.) that reference the target symbol.

        :param relative_file_path: The relative path to the file.
        :param line: The 0-indexed line number.
        :param column: The 0-indexed column number.
        :param include_imports: whether to also include imports as references.
            Unfortunately, the LSP does not have an import type, so the references corresponding to imports
            will not be easily distinguishable from definitions.
        :param include_self: whether to include the references that is the "input symbol" itself. 
            Only has an effect if the relative_file_path, line and column point to a symbol, for example a definition.
        :return: List of symbols that reference the target symbol.
        """
        if not self.server_started:
            self.logger.log(
                "request_referencing_symbols called before Language Server started",
                logging.ERROR,
            )
            raise MultilspyException("Language Server not started")

        # First, get all references to the symbol
        references = await self.request_references(relative_file_path, line, column)
        if not references:
            return []

        # For each reference, find the containing symbol
        result = []
        incoming_symbol = None
        for ref in references:
            ref_path = ref["relativePath"]
            ref_line = ref["range"]["start"]["line"]
            ref_col = ref["range"]["start"]["character"]

            with self.open_file(ref_path) as file_data:
                # Get the containing symbol for this reference
                containing_symbol = await self.request_containing_symbol(
                    ref_path, ref_line, ref_col
                )
                if containing_symbol is None:
                    # TODO: HORRIBLE HACK! I don't know how to do it better for now...
                    # THIS IS BOUND TO BREAK IN MANY CASES! IT IS ALSO SPECIFIC TO PYTHON!
                    # Background:
                    # When a variable is used to change something, like 
                    #
                    # instance = MyClass()
                    # instance.status = "new status"
                    #
                    # we can't find the containing symbol for the reference to `status`
                    # since there is no container on the line of the reference
                    # The hack is to try to find a variable symbol in the containing module
                    # by using the text of the reference to find the variable name (In a very heuristic way)
                    # and then look for a symbol with that name and kind Variable
                    ref_text = file_data.contents.split("\n")[ref_line]
                    if "." in ref_text:   
                        containing_symbol_name = ref_text.split(".")[0]
                        all_symbols, _ = await self.request_document_symbols(ref_path)
                        for symbol in all_symbols:
                            if symbol["name"] == containing_symbol_name and symbol["kind"] == multilspy_types.SymbolKind.Variable:
                                containing_symbol = copy(symbol)
                                containing_symbol["location"] = ref
                                containing_symbol["range"] = ref["range"]
                                break
                if containing_symbol is None:
                    self.logger.log(f"Could not find containing symbol for {ref_path}:{ref_line}:{ref_col}", logging.WARNING)
                    continue

                # Checking for self-reference
                if (
                    containing_symbol["location"]["relativePath"] == relative_file_path
                    and containing_symbol["selectionRange"]["start"]["line"] == ref_line
                    and containing_symbol["selectionRange"]["start"]["character"] == ref_col
                ):
                    incoming_symbol = containing_symbol
                    if include_self:
                        result.append(containing_symbol)
                        continue
                    else:
                        self.logger.log(f"Found self-reference for {incoming_symbol['name']}, skipping it since {include_self=}", logging.DEBUG)
                        continue
                
                # checking whether reference is an import
                # This is neither really safe nor elegant, but if we don't do it,
                # there is no way to distinguish between definitions and imports as import is not a symbol-type
                # and we get the type referenced symbol resulting from imports...
                if (not include_imports \
                    and incoming_symbol is not None \
                    and containing_symbol["name"] == incoming_symbol["name"] \
                    and containing_symbol["kind"] == incoming_symbol["kind"] \
                ):
                    self.logger.log(
                        f"Found import of referenced symbol {incoming_symbol['name']}" 
                        f"in {containing_symbol['location']['relativePath']}, skipping",
                        logging.DEBUG
                    )
                    continue
                
                result.append(containing_symbol)

        return result
    
    async def request_containing_symbol(
        self,
        relative_file_path: str,
        line: int,
        column: Optional[int] = None,
        strict: bool = False,
    ) -> multilspy_types.UnifiedSymbolInformation | None:
        """
        Finds the first symbol containing the position for the given file.
        For Python, container symbols are considered to be those with kinds corresponding to
        functions, methods, or classes (typically: Function (12), Method (6), Class (5)).

        The method operates as follows:
          - Request the document symbols for the file.
          - Filter symbols to those that start at or before the given line.
          - From these, first look for symbols whose range contains the (line, column).
          - If one or more symbols contain the position, return the one with the greatest starting position
            (i.e. the innermost container).
          - If none (strictly) contain the position, return the symbol with the greatest starting position
            among those above the given line.
          - If no container candidates are found, return None.

        :param relative_file_path: The relative path to the Python file.
        :param line: The 0-indexed line number.
        :param column: The 0-indexed column (also called character). If not passed, the lookup will be based
            only on the line.
        :param strict: If True, the position must be strictly within the range of the symbol.
            Setting to True is useful for example for finding the parent of a symbol, as with strict=False,
            and the line pointing to a symbol itself, the containing symbol will be the symbol itself
            (and not the parent).
        :return: The container symbol (if found) or None.
        """
        # checking if the line is empty, unfortunately ugly and duplicating code, but I don't want to refactor
        with self.open_file(relative_file_path):
            absolute_file_path = str(
                PurePath(self.repository_root_path, relative_file_path)
            )
            content = FileUtils.read_file(self.logger, absolute_file_path)
            if content.split("\n")[line].strip() == "":
                self.logger.log(
                    f"Passing empty lines to request_container_symbol is currently not supported, {relative_file_path=}, {line=}",
                    logging.ERROR,
                )
                return

        symbols, _ = await self.request_document_symbols(relative_file_path)
        
        # make jedi and pyright api compatible
        # the former has no location, the later has no range
        # we will just always add location of the desired format to all symbols
        for symbol in symbols:
            if "location" not in symbol:
                range = symbol["range"]
                location = multilspy_types.Location(
                    uri=f"file:/{absolute_file_path}",
                    range=range,
                    absolutePath=absolute_file_path,
                    relativePath=relative_file_path,
                )
                symbol["location"] = location
            else:
                location = symbol["location"]
                assert "range" in location
                location["absolutePath"] = absolute_file_path
                location["relativePath"] = relative_file_path
                location["uri"] = f"file:/{absolute_file_path}"

        # Allowed container kinds, currently only for Python
        container_symbol_kinds = {
            multilspy_types.SymbolKind.Method, 
            multilspy_types.SymbolKind.Function, 
            multilspy_types.SymbolKind.Class
        }

        def is_position_in_range(line: int, range_d: multilspy_types.Range) -> bool:
            start = range_d["start"]
            end = range_d["end"]

            column_condition = True
            if strict:
                line_condition = end["line"] >= line > start["line"]
                if column is not None:
                    column_condition = column > start["character"]
            else:
                line_condition = end["line"] >= line >= start["line"]
                if column is not None:
                    column_condition = column >= start["character"]
            return line_condition and column_condition

        # Only consider containers that are not one-liners (otherwise we may get imports)
        candidate_containers = [
            s for s in symbols if s["kind"] in container_symbol_kinds and s["location"]["range"]["start"]["line"] != s["location"]["range"]["end"]["line"]
        ]
        var_containers = [
            s for s in symbols if s["kind"] == multilspy_types.SymbolKind.Variable
        ]
        candidate_containers.extend(var_containers)
        
        if not candidate_containers:
            return None

        # From the candidates, find those whose range contains the given position.
        containing_symbols = []
        for symbol in candidate_containers:
            s_range = symbol["location"]["range"]
            if not is_position_in_range(line, s_range):
                continue
            containing_symbols.append(symbol)

        if containing_symbols:
            # Return the one with the greatest starting position (i.e. the innermost container).
            return max(containing_symbols, key=lambda s: s["location"]["range"]["start"]["line"])
        else:
            return None
    
    async def request_container_of_symbol(self, symbol: multilspy_types.UnifiedSymbolInformation) -> multilspy_types.UnifiedSymbolInformation | None:
        """
        Finds the container of the given symbol if there is one.
        """
        assert "location" in symbol
        return await self.request_containing_symbol(
            symbol["location"]["relativePath"],
            symbol["location"]["range"]["start"]["line"],
            symbol["location"]["range"]["start"]["character"],
            strict=True,
        )
    
    async def request_defining_symbol(
        self,
        relative_file_path: str,
        line: int,
        column: int,
    ) -> Optional[multilspy_types.UnifiedSymbolInformation]:
        """
        Finds the symbol that defines the symbol at the given location.
        
        This method first finds the definition of the symbol at the given position,
        then retrieves the full symbol information for that definition.
        
        :param relative_file_path: The relative path to the file.
        :param line: The 0-indexed line number.
        :param column: The 0-indexed column number.
        :return: The symbol information for the definition, or None if not found.
        """
        if not self.server_started:
            self.logger.log(
                "request_defining_symbol called before Language Server started",
                logging.ERROR,
            )
            raise MultilspyException("Language Server not started")
            
        # Get the definition location(s)
        definitions = await self.request_definition(relative_file_path, line, column)
        if not definitions:
            return None
        
        # Use the first definition location
        definition = definitions[0]
        def_path = definition["relativePath"]
        def_line = definition["range"]["start"]["line"]
        def_col = definition["range"]["start"]["character"]
        
        # Find the symbol at or containing this location
        defining_symbol = await self.request_containing_symbol(
            def_path, def_line, def_col, strict=False
        )
        
        return defining_symbol
    
    @property
    def _cache_path(self) -> Path:
        return Path(self.repository_root_path) / ".multilspy" / "cache" / "document_symbols_cache.pkl"
    
    def save_cache(self):
        if self._cache_has_changed:
            self.logger.log(f"Saving updated document symbols cache to {self._cache_path}", logging.INFO)
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._cache_path, "wb") as f:
                pickle.dump(self._document_symbols_cache, f)
            
    def load_cache(self):
        if not self._cache_path.exists():
            return
        self.logger.log(f"Loading document symbols cache from {self._cache_path}", logging.INFO)
        with open(self._cache_path, "rb") as f:
            self._document_symbols_cache = pickle.load(f)


@ensure_all_methods_implemented(LanguageServer)
class SyncLanguageServer:
    """
    The SyncLanguageServer class provides a language agnostic interface to the Language Server Protocol.
    It is used to communicate with Language Servers of different programming languages.
    """

    def __init__(self, language_server: LanguageServer) -> None:
        self.language_server = language_server
        self.loop = None
        self.loop_thread = None
        
        self._server_context = None

    @classmethod
    def create(
        cls, config: MultilspyConfig, logger: MultilspyLogger, repository_root_path: str
    ) -> "SyncLanguageServer":
        """
        Creates a language specific LanguageServer instance based on the given configuration, and appropriate settings for the programming language.

        If language is Java, then ensure that jdk-17.0.6 or higher is installed, `java` is in PATH, and JAVA_HOME is set to the installation directory.

        :param repository_root_path: The root path of the repository.
        :param config: The Multilspy configuration.
        :param logger: The logger to use.

        :return SyncLanguageServer: A language specific LanguageServer instance.
        """
        return SyncLanguageServer(LanguageServer.create(config, logger, repository_root_path))

    @contextmanager
    def open_file(self, relative_file_path: str) -> Iterator[None]:
        """
        Open a file in the Language Server. This is required before making any requests to the Language Server.

        :param relative_file_path: The relative path of the file to open.
        """
        with self.language_server.open_file(relative_file_path):
            yield

    def insert_text_at_position(
        self, relative_file_path: str, line: int, column: int, text_to_be_inserted: str
    ) -> multilspy_types.Position:
        """
        Insert text at the given line and column in the given file and return 
        the updated cursor position after inserting the text.

        :param relative_file_path: The relative path of the file to open.
        :param line: The line number at which text should be inserted.
        :param column: The column number at which text should be inserted.
        :param text_to_be_inserted: The text to insert.
        """
        return self.language_server.insert_text_at_position(relative_file_path, line, column, text_to_be_inserted)

    def delete_text_between_positions(
        self,
        relative_file_path: str,
        start: multilspy_types.Position,
        end: multilspy_types.Position,
    ) -> str:
        """
        Delete text between the given start and end positions in the given file and return the deleted text.
        """
        return self.language_server.delete_text_between_positions(relative_file_path, start, end)

    def get_open_file_text(self, relative_file_path: str) -> str:
        """
        Get the contents of the given opened file as per the Language Server.

        :param relative_file_path: The relative path of the file to open.
        """
        return self.language_server.get_open_file_text(relative_file_path)
   
    @contextmanager
    def start_server(self) -> Iterator["SyncLanguageServer"]:
        """
        Starts the language server process and connects to it.

        :return: None
        """
        self.loop = asyncio.new_event_loop()
        loop_thread = threading.Thread(target=self.loop.run_forever, daemon=True)
        loop_thread.start()
        ctx = self.language_server.start_server()
        asyncio.run_coroutine_threadsafe(ctx.__aenter__(), loop=self.loop).result()
        yield self
        asyncio.run_coroutine_threadsafe(ctx.__aexit__(None, None, None), loop=self.loop).result()
        self.loop.call_soon_threadsafe(self.loop.stop)
        loop_thread.join()

    def request_definition(self, file_path: str, line: int, column: int) -> List[multilspy_types.Location]:
        """
        Raise a [textDocument/definition](https://microsoft.github.io/language-server-protocol/specifications/lsp/3.17/specification/#textDocument_definition) request to the Language Server
        for the symbol at the given line and column in the given file. Wait for the response and return the result.

        :param relative_file_path: The relative path of the file that has the symbol for which definition should be looked up
        :param line: The line number of the symbol
        :param column: The column number of the symbol

        :return List[multilspy_types.Location]: A list of locations where the symbol is defined
        """
        result = asyncio.run_coroutine_threadsafe(
            self.language_server.request_definition(file_path, line, column), self.loop
        ).result()
        return result

    def request_references(self, file_path: str, line: int, column: int) -> List[multilspy_types.Location]:
        """
        Raise a [textDocument/references](https://microsoft.github.io/language-server-protocol/specifications/lsp/3.17/specification/#textDocument_references) request to the Language Server
        to find references to the symbol at the given line and column in the given file. Wait for the response and return the result.

        :param relative_file_path: The relative path of the file that has the symbol for which references should be looked up
        :param line: The line number of the symbol
        :param column: The column number of the symbol

        :return List[multilspy_types.Location]: A list of locations where the symbol is referenced
        """
        result = asyncio.run_coroutine_threadsafe(
            self.language_server.request_references(file_path, line, column), self.loop
        ).result()
        return result

    def request_completions(
        self, relative_file_path: str, line: int, column: int, allow_incomplete: bool = False
    ) -> List[multilspy_types.CompletionItem]:
        """
        Raise a [textDocument/completion](https://microsoft.github.io/language-server-protocol/specifications/lsp/3.17/specification/#textDocument_completion) request to the Language Server
        to find completions at the given line and column in the given file. Wait for the response and return the result.

        :param relative_file_path: The relative path of the file that has the symbol for which completions should be looked up
        :param line: The line number of the symbol
        :param column: The column number of the symbol

        :return List[multilspy_types.CompletionItem]: A list of completions
        """
        result = asyncio.run_coroutine_threadsafe(
            self.language_server.request_completions(relative_file_path, line, column, allow_incomplete),
            self.loop,
        ).result()
        return result

    def request_document_symbols(self, relative_file_path: str) -> Tuple[List[multilspy_types.UnifiedSymbolInformation], Union[List[multilspy_types.TreeRepr], None]]:
        """
        Raise a [textDocument/documentSymbol](https://microsoft.github.io/language-server-protocol/specifications/lsp/3.17/specification/#textDocument_documentSymbol) request to the Language Server
        to find symbols in the given file. Wait for the response and return the result.

        :param relative_file_path: The relative path of the file that has the symbols

        :return Tuple[List[multilspy_types.UnifiedSymbolInformation], Union[List[multilspy_types.TreeRepr], None]]: A list of symbols in the file, and the tree representation of the symbols
        """
        result = asyncio.run_coroutine_threadsafe(
            self.language_server.request_document_symbols(relative_file_path), self.loop
        ).result()
        return result

    def request_hover(self, relative_file_path: str, line: int, column: int) -> Union[multilspy_types.Hover, None]:
        """
        Raise a [textDocument/hover](https://microsoft.github.io/language-server-protocol/specifications/lsp/3.17/specification/#textDocument_hover) request to the Language Server
        to find the hover information at the given line and column in the given file. Wait for the response and return the result.

        :param relative_file_path: The relative path of the file that has the hover information
        :param line: The line number of the symbol
        :param column: The column number of the symbol

        :return None
        """
        result = asyncio.run_coroutine_threadsafe(
            self.language_server.request_hover(relative_file_path, line, column), self.loop
        ).result()
        return result
    
    # ----------------------------- FROM HERE ON MODIFICATIONS BY MISCHA --------------------
    
    def request_parsed_files(self) -> list[str]:
        """This is slow, as it finds all files by finding all symbols. 
        
        This seems to be the only way, the LSP does not provide any endpoints for listing project files."""
        assert self.loop
        result = asyncio.run_coroutine_threadsafe(
            self.language_server.request_parsed_files(), self.loop
        ).result()
        return result
    
    def request_referencing_symbols(
        self, relative_file_path: str, line: int, column: int,
        include_imports: bool = True, include_self: bool = False,
    ) -> List[multilspy_types.UnifiedSymbolInformation]:
        """
        Finds all symbols that reference the symbol at the given location.
        This is similar to request_references but filters to only include symbols
        (functions, methods, classes, etc.) that reference the target symbol.

        :param relative_file_path: The relative path to the file.
        :param line: The 0-indexed line number.
        :param column: The 0-indexed column number.
        :param include_imports: whether to also include imports as references.
            Unfortunately, the LSP does not have an import type, so the references corresponding to imports
            will not be easily distinguishable from definitions.
        :param include_self: whether to include the references that is the "input symbol" itself. 
            Only has an effect if the relative_file_path, line and column point to a symbol, for example a definition.
        :return: List of symbols that reference the target symbol.
        """
        assert self.loop
        result = asyncio.run_coroutine_threadsafe(
            self.language_server.request_referencing_symbols(
                relative_file_path, 
                line, 
                column, 
                include_imports=include_imports, 
                include_self=include_self
            ), 
            self.loop
        ).result()
        return result
        
    def request_containing_symbol(
        self, relative_file_path: str, line: int, 
        column: Optional[int] = None, strict: bool = False
    ) -> multilspy_types.UnifiedSymbolInformation | None:
        """
        Finds the first symbol containing the position for the given file.
        For Python, container symbols are considered to be those with kinds corresponding to
        functions, methods, or classes (typically: Function (12), Method (6), Class (5)).
        
        The method operates as follows:
          - Request the document symbols for the file.
          - Filter symbols to those that start at or before the given line.
          - From these, first look for symbols whose range contains the (line, column).
          - If one or more symbols contain the position, return the one with the greatest starting position
            (i.e. the innermost container).
          - If none (strictly) contain the position, return the symbol with the greatest starting position 
            among those above the given line.
          - If no container candidates are found, return None.
        
        :param relative_file_path: The relative path to the Python file.
        :param line: The 0-indexed line number.
        :param column: The 0-indexed column (also called character). If not passed, the lookup will be based
            only on the line.
        :param strict: If True, the position must be strictly within the range of the symbol.
            Setting to true is useful for example for finding the parent of a symbol, as with strict=False,
            and the line pointing to a symbol itself, the containing symbol will be the symbol itself 
            (and not the parent).
        :return: The container symbol (if found) or None.
        """
        assert self.loop
        result = asyncio.run_coroutine_threadsafe(
            self.language_server.request_containing_symbol(relative_file_path, line, column=column, strict=strict), self.loop
        ).result()
        return result
    
    def request_container_of_symbol(self, symbol: multilspy_types.UnifiedSymbolInformation) -> multilspy_types.UnifiedSymbolInformation | None:
        """
        Finds the container of the given symbol if there is one.
        """
        assert self.loop
        result = asyncio.run_coroutine_threadsafe(
            self.language_server.request_container_of_symbol(symbol), self.loop
        ).result()
        return result
    
    def request_defining_symbol(
        self, relative_file_path: str, line: int, column: int
    ) -> Optional[multilspy_types.UnifiedSymbolInformation]:
        """
        Finds the symbol that defines the symbol at the given location.
        
        This method first finds the definition of the symbol at the given position,
        then retrieves the full symbol information for that definition.
        
        :param relative_file_path: The relative path to the file.
        :param line: The 0-indexed line number.
        :param column: The 0-indexed column number.
        :return: The symbol information for the definition, or None if not found.
        """
        assert self.loop
        result = asyncio.run_coroutine_threadsafe(
            self.language_server.request_defining_symbol(relative_file_path, line, column), self.loop
        ).result()
        return result
    
    def start(self) -> "SyncLanguageServer":
        """
        Starts the language server process and connects to it. Call shutdown when ready.

        :return: self for method chaining
        """
        self.loop = asyncio.new_event_loop()
        self.loop_thread = threading.Thread(target=self.loop.run_forever, daemon=True)
        self.loop_thread.start()
        self._server_context = self.language_server.start_server()
        asyncio.run_coroutine_threadsafe(self._server_context.__aenter__(), loop=self.loop).result()
        return self

    def stop(self) -> None:
        """
        Shuts down the language server process and cleans up resources.
        Must be called after start().
        """
        if not self.loop or not self.loop_thread:
            raise MultilspyException("Language Server not started")
            
        asyncio.run_coroutine_threadsafe(self._server_context.__aexit__(None, None, None), loop=self.loop).result()
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.loop_thread.join()
        self.loop = None
        self.loop_thread = None
        self.save_cache()
        
    def save_cache(self):
        """
        Save the cache to a file.
        """
        self.language_server.save_cache()
    
    def load_cache(self):
        """
        Load the cache from a file.
        """
        self.language_server.load_cache()
    
        
    def __del__(self):
        if self.loop:
            self.stop()