"""Canonical input counts are read-only history metadata, not model work."""
import json,unittest
from serve import test_cow_workspace as fixture
from serve import test_cow_workspace_http as http
class HistoryCountTests(unittest.TestCase):
 def setUp(self):fixture.CowWorkspaceTests.setUp(self)
 def test_root_and_tip_counts_follow_canonical_ancestry(self):
  root=fixture.CowWorkspaceTests._root(self,tokens=(1,2))
  first=self.workspace.history('chat');self.assertEqual(first['root_input_cells'],2);self.assertEqual(first['tip_input_cells'],2)
  target,_,_=fixture.CowWorkspaceTests._answer(self,root,output=(5,6),status='stopped')
  before=self.store.presentation_catalog(self.ns);value=self.workspace.history('chat')
  self.assertEqual(value['root_input_cells'],2);self.assertEqual(value['tip_input_cells'],4)
  self.assertEqual(self.store.presentation_catalog(self.ns),before)
  head=self.store.head('chat')
  self.store.fork('sibling',root,namespace_id=self.ns,operation_id='sibling-fork',source_branch=head.name,source_revision=head.revision,source_tip=head.tip)
  self.assertEqual(self.workspace.history('sibling')['tip_input_cells'],2)
class HistoryCountHTTPTests(unittest.TestCase):
 def setUp(self):http.WorkspaceHTTPTests.setUp(self)
 close_server=http.WorkspaceHTTPTests.close_server
 call=http.WorkspaceHTTPTests.call
 namespace=http.WorkspaceHTTPTests.namespace
 def test_counts_come_from_actual_http_without_model_work(self):
  ns=self.namespace();code,raw=self.call('/cow/text/prepare',dict(branch='chat',messages=[dict(role='user',content='hello')],namespace_id=ns,operation_id='count-prepare',stream=False));self.assertEqual(code,200)
  head=self.coordinator.store.head('chat');expected=len(self.coordinator.store.lineage(head.root)[1]);before=dict(self.models[0].stats)
  code,raw=self.call('/cow/workspace/history?branch=chat');self.assertEqual(code,200);value=json.loads(raw)
  self.assertEqual(value['root_input_cells'],expected);self.assertEqual(value['tip_input_cells'],expected);self.assertEqual(self.models[0].stats,before)

class MixedHistoryCountTests(unittest.TestCase):
 def test_image_rows_and_text_tokens_are_counted_as_model_input_cells(self):
  from serve.test_cow_vision_text_flow import CowVisionTextFlowTests
  case=CowVisionTextFlowTests();case.setUp();self.addCleanup(case.doCleanups)
  root=case.prepare();history=case.models[-1].history(root['snapshot'])
  expected=sum(len(span['tokens']) if span['kind']=='text' else span['rows'] for span in history.spans)
  self.assertTrue(any(span['kind']=='image' for span in history.spans))
  case.text.answer('main',1,'count-answer',2).durable.result(5)
  before=dict(case.models[-1].backend.stats);view=case.workspace.history('main')
  self.assertEqual(view['root_input_cells'],expected);self.assertEqual(view['tip_input_cells'],expected+2)
  self.assertEqual(case.models[-1].backend.stats,before)
