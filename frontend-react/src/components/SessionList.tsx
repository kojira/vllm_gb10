import { useState, useEffect } from 'react'
import {
  Drawer,
  Box,
  List,
  ListItem,
  ListItemButton,
  ListItemText,
  ListItemSecondaryAction,
  IconButton,
  Typography,
  Button,
  Divider,
} from '@mui/material'
import AddIcon from '@mui/icons-material/Add'
import DeleteIcon from '@mui/icons-material/Delete'
import { Session } from '../types'
import { fetchSessions, deleteSession } from '../services/api'

interface SessionListProps {
  open: boolean
  onClose: () => void
  currentSessionId: string | null
  onSelectSession: (session: Session) => void
  onNewSession: () => void
}

export function SessionList({
  open,
  onClose,
  currentSessionId,
  onSelectSession,
  onNewSession,
}: SessionListProps) {
  const [sessions, setSessions] = useState<Session[]>([])

  useEffect(() => {
    if (open) {
      fetchSessions().then(setSessions).catch(console.error)
    }
  }, [open])

  const handleDelete = async (id: string, e: React.MouseEvent) => {
    e.stopPropagation()
    if (confirm('この会話を削除しますか？')) {
      await deleteSession(id)
      setSessions(sessions.filter((s) => s.id !== id))
    }
  }

  const formatDate = (dateStr: string) => {
    const date = new Date(dateStr)
    return date.toLocaleDateString('ja-JP', {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    })
  }

  return (
    <Drawer anchor="left" open={open} onClose={onClose}>
      <Box sx={{ width: 320, p: 2 }}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
          <Typography variant="h6">会話履歴</Typography>
          <Button
            startIcon={<AddIcon />}
            variant="contained"
            size="small"
            onClick={() => {
              onNewSession()
              onClose()
            }}
          >
            新しい会話
          </Button>
        </Box>
        <Divider sx={{ mb: 2 }} />
        <List>
          {sessions.length === 0 ? (
            <Typography color="text.secondary" sx={{ textAlign: 'center', py: 4 }}>
              会話履歴がありません
            </Typography>
          ) : (
            sessions.map((session) => (
              <ListItem key={session.id} disablePadding>
                <ListItemButton
                  selected={session.id === currentSessionId}
                  onClick={() => {
                    onSelectSession(session)
                    onClose()
                  }}
                >
                  <ListItemText
                    primary={session.title || '無題の会話'}
                    secondary={
                      <>
                        {session.engine || '-'} | {formatDate(session.updated_at)}
                      </>
                    }
                    primaryTypographyProps={{
                      noWrap: true,
                      sx: { maxWidth: 200 },
                    }}
                  />
                  <ListItemSecondaryAction>
                    <IconButton
                      edge="end"
                      size="small"
                      onClick={(e) => handleDelete(session.id, e)}
                    >
                      <DeleteIcon fontSize="small" />
                    </IconButton>
                  </ListItemSecondaryAction>
                </ListItemButton>
              </ListItem>
            ))
          )}
        </List>
      </Box>
    </Drawer>
  )
}

